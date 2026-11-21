//! Token generation for `extractor` sessions (design 04-generation.md):
//! greedy decode with kv-cache, plus schema-constrained filling where a
//! derived json state machine forces the structure and masks the model's
//! choices to valid continuations. The wrapper knows tokens and logits
//! only — prompt rendering, validation, caching live in modelcaller.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array2, Array4, Ix4};
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;

use libtokenizer::TokenizerHandle;

use crate::session::LoadedSession;

type CmdErr = (String, String);

pub struct GenOutput {
    pub text: String,
    pub n_new_tokens: usize,
    pub finished: bool,
}

// ---------------------------------------------------------------------------
// token classification (computed once per session, cached on it)
// ---------------------------------------------------------------------------

/// Category sets over the vocabulary, so per-step logit masks are set
/// lookups instead of rescanning ~151k token strings each step.
pub struct TokenClass {
    /// decoded text per id (`" 1"` for the byte-BPE `Ġ1`)
    text: Vec<String>,
    /// ascii digits only — continue a numeric literal
    digit: Vec<u32>,
    /// `,"` — numeric terminal: separator + next key's opening quote
    sep_quote: Vec<u32>,
    /// `}` / ` }` — object close
    brace: Vec<u32>,
    /// `]` / ` ]` — array close
    bracket: Vec<u32>,
    /// `"` — string close
    quote: Vec<u32>,
    /// bare `,` — list element separator
    comma: Vec<u32>,
    /// no `"`, `\`, newline — free string content
    str_safe: Vec<u32>,
}

impl TokenClass {
    fn build(tok: &TokenizerHandle) -> Self {
        let n = tok.vocab_size();
        let mut text = Vec::with_capacity(n);
        let (mut digit, mut sep_quote) = (Vec::new(), Vec::new());
        let (mut brace, mut bracket, mut quote, mut comma) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut str_safe = Vec::new();
        for id in 0..n as u32 {
            // decode() applies the tokenizer's decoder, so BPE `Ġ` and
            // WordPiece `##` markers come out as real text; the `##` marker
            // itself survives a single-id decode and is stripped here
            let raw = tok
                .decode(&[id], false)
                .unwrap_or_else(|_| tok.id_to_token(id).unwrap_or_default());
            let stripped = raw.strip_prefix("##").unwrap_or(&raw);
            let decoded = stripped.replace('Ġ', " ");
            let is_special = decoded.contains('<') && decoded.contains('>');
            text.push(decoded.clone());
            if is_special || decoded.is_empty() {
                continue;
            }
            if decoded.chars().all(|c| c.is_ascii_digit()) {
                digit.push(id);
            }
            match decoded.as_str() {
                ",\"" => sep_quote.push(id),
                "}" | " }" => brace.push(id),
                "]" | " ]" => bracket.push(id),
                "\"" => quote.push(id),
                "," => comma.push(id),
                _ => {}
            }
            if !decoded.contains('"')
                && !decoded.contains('\\')
                && !decoded.contains('\n')
                && !decoded.contains('\r')
            {
                str_safe.push(id);
            }
        }
        Self {
            text,
            digit,
            sep_quote,
            brace,
            bracket,
            quote,
            comma,
            str_safe,
        }
    }

    fn decoded(&self, id: u32) -> &str {
        self.text.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// constrained fill: schema-derived state machine (design §4.4)
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Deserialize)]
struct SchemaFieldDef {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    required: bool,
    #[serde(default, rename = "enum")]
    enum_values: Option<Vec<String>>,
}

/// What the machine wants next. `ForceIds` carries the decoded text so the
/// output string never re-encodes; `Decide` masks one greedy pick.
#[derive(Debug)]
enum Action {
    ForceIds(Vec<u32>, String),
    Decide(Vec<u32>),
    Finish,
}

#[derive(Debug, Clone, PartialEq)]
enum State {
    /// model picks: start the next field's fragment vs close the object
    /// (close only offered once every remaining field is optional)
    FieldBoundary {
        next: usize,
        sep_emitted: bool,
    },
    Numeric {
        idx: usize,
        digits: usize,
    },
    /// free string content (opening quote already emitted)
    StrContent {
        idx: usize,
    },
    /// literal candidates (bool / enum) filtered to those matching so far
    Literal {
        idx: usize,
        cands: Vec<Vec<u32>>,
        depth: usize,
    },
    /// machine is about to force the `[` opening a list value
    ListOpen {
        idx: usize,
    },
    /// after `[`: model picks an element's opening quote vs `]` (empty list)
    ListFirst {
        idx: usize,
    },
    /// inside a list element string (`items` = closed elements so far)
    ListElem {
        idx: usize,
        items: usize,
    },
    /// after an element close: `,` more vs `]`
    ListBoundary {
        idx: usize,
        items: usize,
    },
    /// machine is about to force ` "` opening the next list element
    ListAfterComma {
        idx: usize,
        items: usize,
    },
    Done,
}

/// Builds the constrained json object one decision at a time. Owns the
/// output text (`text()`); the decode loop never appends to it directly.
pub struct FillMachine {
    fields: Vec<SchemaFieldDef>,
    state: State,
    out: String,
    max_items: usize,
    max_num_digits: usize,
}

impl FillMachine {
    pub fn from_schema_json(schema: &Value) -> Result<Self, CmdErr> {
        let fields: Vec<SchemaFieldDef> = serde_json::from_value(
            schema
                .get("fields")
                .cloned()
                .unwrap_or_else(|| Value::Array(vec![])),
        )
        .map_err(|e| ("bad_req".to_string(), format!("schema.fields: {e}")))?;
        if fields.len() > 12 {
            return Err(("bad_req".into(), "at most 12 schema fields".into()));
        }
        for f in &fields {
            if !matches!(
                f.kind.as_str(),
                "numeric" | "string" | "bool" | "list_string"
            ) {
                return Err((
                    "bad_req".into(),
                    format!("field '{}' has unsupported type '{}'", f.name, f.kind),
                ));
            }
            if f.name.is_empty() || f.name.len() > 64 {
                return Err((
                    "bad_req".into(),
                    format!("bad field name length: '{}'", f.name),
                ));
            }
        }
        Ok(Self {
            fields,
            state: State::FieldBoundary {
                next: 0,
                sep_emitted: false,
            },
            out: "{".to_string(),
            max_items: 16,
            max_num_digits: 12,
        })
    }

    pub fn text(&self) -> &str {
        &self.out
    }

    /// plain string values open with a quote (lists open with `[`, forced)
    fn value_open_quote(&self, idx: usize) -> bool {
        self.fields[idx].kind == "string"
    }

    /// the fragment emitted when the model opts to fill field `idx`. A
    /// comma is owed unless the numeric terminal `,"` already emitted it.
    fn field_fragment(&self, idx: usize, sep_emitted: bool) -> String {
        let f = &self.fields[idx];
        let mut s = String::new();
        if sep_emitted {
            s.push_str(&f.name);
            s.push_str("\": ");
        } else {
            if idx > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(&f.name);
            s.push_str("\": ");
        }
        if self.value_open_quote(idx) {
            s.push('"');
        }
        s
    }

    fn literal_candidates(&self, idx: usize, tok: &TokenizerHandle) -> Vec<Vec<u32>> {
        let f = &self.fields[idx];
        let lits: Vec<String> = if f.kind == "bool" {
            vec!["true".into(), "false".into()]
        } else {
            f.enum_values
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|v| format!("{v}\""))
                .collect()
        };
        lits.iter()
            .filter_map(|l| tok.encode(l, false).ok().map(|e| e.ids))
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn next_action(&self, tc: &TokenClass, tok: &TokenizerHandle) -> Result<Action, CmdErr> {
        match &self.state {
            State::Done => Ok(Action::Finish),
            State::FieldBoundary { next, sep_emitted } => {
                if *next >= self.fields.len() {
                    return Ok(Action::ForceIds(Vec::new(), "}".to_string()));
                }
                let frag = self.field_fragment(*next, *sep_emitted);
                let enc = tok
                    .encode(&frag, false)
                    .map_err(|e| ("tokenize_err".into(), e.to_string()))?;
                let mut allowed: Vec<u32> = enc.ids.first().copied().into_iter().collect();
                let rest_optional = self.fields[*next..].iter().all(|f| !f.required);
                if rest_optional {
                    allowed.extend_from_slice(&tc.brace);
                }
                Ok(Action::Decide(allowed))
            }
            State::Numeric { idx, digits } => {
                if *digits == 0 {
                    return Ok(Action::Decide(tc.digit.clone()));
                }
                if *digits >= self.max_num_digits {
                    let mut allowed = tc.brace.clone();
                    if *idx + 1 < self.fields.len() {
                        allowed.extend_from_slice(&tc.sep_quote);
                    }
                    return Ok(Action::Decide(allowed));
                }
                let mut allowed = tc.digit.clone();
                if *idx + 1 < self.fields.len() {
                    allowed.extend_from_slice(&tc.sep_quote);
                }
                allowed.extend_from_slice(&tc.brace);
                Ok(Action::Decide(allowed))
            }
            State::StrContent { .. } | State::ListElem { .. } => {
                let mut allowed = tc.str_safe.clone();
                allowed.extend_from_slice(&tc.quote);
                Ok(Action::Decide(allowed))
            }
            State::Literal { cands, depth, .. } => {
                let nexts: Vec<u32> = cands
                    .iter()
                    .filter_map(|s| s.get(*depth).copied())
                    .collect();
                let mut uniq = nexts.clone();
                uniq.sort_unstable();
                uniq.dedup();
                if uniq.is_empty() {
                    // degenerate: no candidate survives — close the string
                    return Ok(Action::Decide(tc.quote.clone()));
                }
                if uniq.len() == 1 {
                    let text = tc.decoded(uniq[0]).to_string();
                    return Ok(Action::ForceIds(vec![uniq[0]], text));
                }
                Ok(Action::Decide(uniq))
            }
            State::ListOpen { .. } => {
                let enc = tok
                    .encode("[", false)
                    .map_err(|e| ("tokenize_err".into(), e.to_string()))?;
                let text = tc
                    .decoded(*enc.ids.first().unwrap_or(&u32::MAX))
                    .to_string();
                Ok(Action::ForceIds(
                    enc.ids,
                    if text.is_empty() { "[".into() } else { text },
                ))
            }
            State::ListFirst { .. } => {
                let mut allowed = tc.quote.clone();
                allowed.extend_from_slice(&tc.bracket);
                Ok(Action::Decide(allowed))
            }
            State::ListBoundary { items, .. } => {
                if *items >= self.max_items {
                    return Ok(Action::Decide(tc.bracket.clone()));
                }
                let mut allowed = tc.comma.clone();
                allowed.extend_from_slice(&tc.bracket);
                Ok(Action::Decide(allowed))
            }
            State::ListAfterComma { .. } => {
                let enc = tok
                    .encode(" \"", false)
                    .map_err(|e| ("tokenize_err".into(), e.to_string()))?;
                Ok(Action::ForceIds(enc.ids, " \"".to_string()))
            }
        }
    }

    /// A forced id from `next_action`: advance the state the forced pick
    /// implies. The text was already appended by [`Self::accept_forced`].
    fn advance_forced(&mut self, id: Option<u32>) {
        match (&self.state, id) {
            (_, None) => self.state = State::Done,
            (State::ListOpen { idx }, Some(_)) => {
                self.state = State::ListFirst { idx: *idx };
            }
            (State::ListAfterComma { idx, items }, Some(_)) => {
                self.state = State::ListElem {
                    idx: *idx,
                    items: *items,
                };
            }
            (State::Literal { .. }, Some(id)) => {
                if let State::Literal { cands, depth, .. } = &mut self.state {
                    cands.retain(|s| s.get(*depth).copied() == Some(id));
                    if !cands.is_empty() {
                        *depth += 1;
                        if cands.iter().all(|s| s.len() == *depth) {
                            let idx = match &self.state {
                                State::Literal { idx, .. } => *idx,
                                _ => unreachable!(),
                            };
                            self.state = State::FieldBoundary {
                                next: idx + 1,
                                sep_emitted: false,
                            };
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Machine-driven output (forced continuations): append the text and
    /// advance the state.
    pub fn accept_forced(&mut self, ids: &[u32], text: &str) {
        self.out.push_str(text);
        let first = ids.first().copied();
        self.advance_forced(first);
    }

    /// Apply a model pick; returns extra token ids to feed back for context
    /// (the remainder of a field fragment). The pick's own text is appended
    /// here — the decode loop never writes `out` on the machine path.
    fn consume(
        &mut self,
        id: u32,
        tc: &TokenClass,
        tok: &TokenizerHandle,
    ) -> Result<Vec<u32>, CmdErr> {
        // at a field boundary the emitted fragment already carries the
        // picked token's text as its prefix — appending the pick too would
        // duplicate it
        if !matches!(self.state, State::FieldBoundary { .. }) || tc.brace.contains(&id) {
            self.out.push_str(tc.decoded(id));
        }
        match self.state.clone() {
            State::FieldBoundary { next, sep_emitted } => {
                if tc.brace.contains(&id) {
                    self.state = State::Done;
                    return Ok(Vec::new());
                }
                let frag = self.field_fragment(next, sep_emitted);
                let enc = tok
                    .encode(&frag, false)
                    .map_err(|e| ("tokenize_err".into(), e.to_string()))?;
                let first_text = enc
                    .ids
                    .first()
                    .map(|&f| tc.decoded(f).to_string())
                    .unwrap_or_default();
                let n = first_text.len();
                let rest = if frag.len() > n && frag.is_char_boundary(n) {
                    frag[n..].to_string()
                } else {
                    String::new()
                };
                self.out.push_str(&frag);
                self.state = match self.fields[next].kind.as_str() {
                    "numeric" => State::Numeric {
                        idx: next,
                        digits: 0,
                    },
                    "bool" => State::Literal {
                        idx: next,
                        cands: self.literal_candidates(next, tok),
                        depth: 0,
                    },
                    // enum strings are literals too (closing quote included)
                    "string" if self.fields[next].enum_values.is_some() => State::Literal {
                        idx: next,
                        cands: self.literal_candidates(next, tok),
                        depth: 0,
                    },
                    "string" => State::StrContent { idx: next },
                    _ => State::ListOpen { idx: next },
                };
                let feed = if rest.is_empty() {
                    Vec::new()
                } else {
                    tok.encode(&rest, false)
                        .map_err(|e| ("tokenize_err".into(), e.to_string()))?
                        .ids
                };
                Ok(feed)
            }
            State::Numeric { idx, digits } => {
                if tc.digit.contains(&id) {
                    self.state = State::Numeric {
                        idx,
                        digits: digits + 1,
                    };
                    return Ok(Vec::new());
                }
                if tc.sep_quote.contains(&id) {
                    // `,"`: the comma separates fields AND is the next key's
                    // opening quote
                    self.state = State::FieldBoundary {
                        next: idx + 1,
                        sep_emitted: true,
                    };
                    return Ok(Vec::new());
                }
                if tc.brace.contains(&id) {
                    self.state = State::Done;
                    return Ok(Vec::new());
                }
                Err((
                    "constraint_err".into(),
                    format!("numeric state admitted '{id}'"),
                ))
            }
            State::StrContent { idx } => {
                if tc.quote.contains(&id) {
                    self.state = State::FieldBoundary {
                        next: idx + 1,
                        sep_emitted: false,
                    };
                }
                Ok(Vec::new())
            }
            State::Literal {
                idx,
                mut cands,
                depth,
            } => {
                cands.retain(|s| s.get(depth).copied() == Some(id));
                if cands.is_empty() {
                    return Err((
                        "constraint_err".into(),
                        format!("literal state admitted '{id}' with no candidate"),
                    ));
                }
                if cands.iter().all(|s| s.len() == depth + 1) {
                    self.state = State::FieldBoundary {
                        next: idx + 1,
                        sep_emitted: false,
                    };
                } else {
                    self.state = State::Literal {
                        idx,
                        cands,
                        depth: depth + 1,
                    };
                }
                Ok(Vec::new())
            }
            State::ListFirst { idx } => {
                if tc.bracket.contains(&id) {
                    self.state = State::FieldBoundary {
                        next: idx + 1,
                        sep_emitted: false,
                    };
                } else {
                    self.state = State::ListElem { idx, items: 0 };
                }
                Ok(Vec::new())
            }
            State::ListElem { idx, items } => {
                if tc.quote.contains(&id) {
                    self.state = State::ListBoundary {
                        idx,
                        items: items + 1,
                    };
                }
                Ok(Vec::new())
            }
            State::ListBoundary { idx, items } => {
                if tc.bracket.contains(&id) {
                    self.state = State::FieldBoundary {
                        next: idx + 1,
                        sep_emitted: false,
                    };
                    return Ok(Vec::new());
                }
                if tc.comma.contains(&id) && items < self.max_items {
                    self.state = State::ListAfterComma { idx, items };
                    return Ok(Vec::new());
                }
                Err((
                    "constraint_err".into(),
                    format!("list boundary admitted '{id}'"),
                ))
            }
            State::ListOpen { .. } | State::ListAfterComma { .. } => Ok(Vec::new()),
            State::Done => Ok(Vec::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// the decode loop (kv-cached greedy; design §4.2/§4.4)
// ---------------------------------------------------------------------------

struct Kv {
    base: String,
    key: Array4<f32>,
    value: Array4<f32>,
}

struct PastInput {
    /// full ONNX input name, e.g. `past_key_values.0.decoder.key`
    full: String,
    /// the name with `.key`/`.value` stripped
    base: String,
    is_value: bool,
}

struct IoNames {
    has_position_ids: bool,
    has_cache_position: bool,
    past: Vec<PastInput>,
    /// shape of the empty past-kv tensors fed on prefill;
    /// `[batch, kv_heads, 0, head_dim]` from the model config when known
    empty_kv: [usize; 4],
}

fn discover_io(sess: &Session) -> IoNames {
    let mut io = IoNames {
        has_position_ids: false,
        has_cache_position: false,
        past: Vec::new(),
        empty_kv: [1, 1, 0, 1],
    };
    for i in sess.inputs() {
        let n = i.name().to_string();
        if n == "position_ids" {
            io.has_position_ids = true;
        } else if n == "cache_position" {
            io.has_cache_position = true;
        } else if let Some(s) = n.strip_prefix("past_key_values.") {
            let (base, is_value) = match s.strip_suffix(".value") {
                Some(b) => (b.to_string(), true),
                None => (s.strip_suffix(".key").unwrap_or(s).to_string(), false),
            };
            io.past.push(PastInput {
                full: n,
                base,
                is_value,
            });
        }
    }
    io
}

/// One forward pass: feed `ids` (+ past kv), return last-position logits
/// and the present kv. `total_len` is the sequence length INCLUDING `ids`.
fn forward(
    sess: &mut Session,
    io: &IoNames,
    ids: &[u32],
    total_len: usize,
    past: Vec<Kv>,
) -> Result<(Vec<f32>, Vec<Kv>), CmdErr> {
    let n = ids.len();
    let input_ids = Array2::<i64>::from_shape_vec((1, n), ids.iter().map(|&t| t as i64).collect())
        .map_err(|e| ("inference_err".into(), format!("shape error: {e}")))?;
    let attention = Array2::<i64>::ones((1, total_len));
    let pos: Array2<i64> = Array2::from_shape_vec(
        (1, n),
        (total_len - n..total_len).map(|v| v as i64).collect(),
    )
    .map_err(|e| ("inference_err".into(), format!("shape error: {e}")))?;

    let mut inputs: Vec<(
        std::borrow::Cow<'static, str>,
        ort::session::SessionInputValue<'static>,
    )> = Vec::new();
    inputs.push((
        "input_ids".into(),
        Tensor::from_array(input_ids).map_err(tensor_err)?.into(),
    ));
    inputs.push((
        "attention_mask".into(),
        Tensor::from_array(attention).map_err(tensor_err)?.into(),
    ));
    if io.has_position_ids {
        inputs.push((
            "position_ids".into(),
            Tensor::from_array(pos.clone()).map_err(tensor_err)?.into(),
        ));
    }
    if io.has_cache_position {
        inputs.push((
            "cache_position".into(),
            Tensor::from_array(pos).map_err(tensor_err)?.into(),
        ));
    }
    let by_base: HashMap<&str, &Kv> = past.iter().map(|k| (k.base.as_str(), k)).collect();
    for p in &io.past {
        // prefill has no past kv; feed a zero-length tensor so the graph's
        // shape ops still see the input (concat along the seq axis is a
        // no-op for it)
        let arr = match by_base.get(p.base.as_str()) {
            Some(kv) => (if p.is_value { &kv.value } else { &kv.key }).clone(),
            None => Array4::zeros(io.empty_kv),
        };
        let t = Tensor::from_array(arr).map_err(tensor_err)?;
        inputs.push((p.full.clone().into(), t.into()));
    }

    let outputs = sess
        .run(inputs)
        .map_err(|e| ("inference_err".into(), format!("inference failed: {e}")))?;

    let mut last_logits = None;
    let mut present: Vec<Kv> = Vec::new();
    for (name, value) in outputs.iter() {
        if name == "logits" {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| ("inference_err".into(), format!("logits extract: {e}")))?;
            let row = match &**shape {
                [v] => data[..*v as usize].to_vec(),
                [1, v] => data[..*v as usize].to_vec(),
                [1, 1, v] => data[..*v as usize].to_vec(),
                [1, l, v] => {
                    let (l, v) = (*l as usize, *v as usize);
                    data[(l - 1) * v..l * v].to_vec()
                }
                s => {
                    return Err((
                        "inference_err".into(),
                        format!("unexpected logits shape {s:?}"),
                    ));
                }
            };
            last_logits = Some(row);
        } else if let Some(suffix) = name.strip_prefix("present.") {
            let arr = value
                .try_extract_array::<f32>()
                .map_err(|e| ("inference_err".into(), format!("kv extract: {e}")))?
                .into_owned();
            let arr = arr
                .into_dimensionality::<Ix4>()
                .map_err(|e| ("inference_err".into(), format!("kv shape: {e}")))?;
            let (base, is_value) = match suffix.strip_suffix(".value") {
                Some(b) => (b.to_string(), true),
                None => (
                    suffix.strip_suffix(".key").unwrap_or(suffix).to_string(),
                    false,
                ),
            };
            match present.iter_mut().find(|k| k.base == base) {
                Some(k) if is_value => k.value = arr,
                Some(k) => k.key = arr,
                None => {
                    let (key, value) = if is_value {
                        (Array4::zeros((1, 1, 1, 1)), arr)
                    } else {
                        (arr, Array4::zeros((1, 1, 1, 1)))
                    };
                    present.push(Kv { base, key, value });
                }
            }
        }
    }
    let logits = last_logits.ok_or_else(|| {
        (
            "inference_err".to_string(),
            "model produced no logits".into(),
        )
    })?;
    Ok((logits, present))
}

fn tensor_err(e: ort::Error) -> CmdErr {
    ("inference_err".into(), format!("tensor error: {e}"))
}

fn argmax_masked(logits: &[f32], allowed: &[u32]) -> Option<u32> {
    let mut best: Option<(f32, u32)> = None;
    for &id in allowed {
        let l = *logits.get(id as usize)?;
        if best.is_none_or(|(b, _)| l > b) {
            best = Some((l, id));
        }
    }
    best.map(|(_, id)| id)
}

/// Greedy generation entry point. `schema` present → constrained fill (the
/// returned text is the machine-built json; unconstrained returns decoded
/// picks).
pub(crate) fn generate(
    loaded: &mut LoadedSession,
    prompt: &str,
    max_new_tokens: usize,
    stop_ids: &[u32],
    stop_strings: &[String],
    schema: Option<&Value>,
) -> Result<GenOutput, CmdErr> {
    if loaded.kind != "extractor" {
        return Err((
            "bad_kind".into(),
            format!("session is a {}, not an extractor", loaded.kind),
        ));
    }
    let tokenizer = loaded.tokenizer.clone();
    let tc = match &loaded.token_class {
        Some(tc) => tc.clone(),
        None => {
            let tc = Arc::new(TokenClass::build(&tokenizer));
            loaded.token_class = Some(tc.clone());
            tc
        }
    };
    let mut machine = match schema {
        Some(s) => Some(FillMachine::from_schema_json(s)?),
        None => None,
    };

    let budget = loaded
        .max_context_window
        .saturating_sub(max_new_tokens)
        .max(64);
    let mut queue = tokenizer
        .encode_truncated(prompt, false, budget)
        .map_err(|e| ("tokenize_err".into(), e.to_string()))?
        .ids;
    if queue.is_empty() {
        return Err(("tokenize_err".into(), "empty prompt".into()));
    }
    let mut total_len = queue.len();
    let mut free_out = String::new();
    let mut n_generated = 0usize;
    let mut finished = false;

    let mut io = discover_io(&loaded.pool[0]);
    if let Some((kv_heads, head_dim)) = loaded.kv_spec {
        io.empty_kv = [1, kv_heads, 0, head_dim];
    }
    log::debug!(
        "generate: prompt_tokens={} pos_ids={} cache_pos={} past_inputs={} empty_kv={:?}",
        queue.len(),
        io.has_position_ids,
        io.has_cache_position,
        io.past.len(),
        io.empty_kv
    );
    let (mut logits, mut past) = forward(&mut loaded.pool[0], &io, &queue, total_len, Vec::new())?;
    queue.clear();

    loop {
        let action = match &mut machine {
            Some(m) => m.next_action(&tc, &tokenizer)?,
            None => Action::Decide(Vec::new()),
        };
        match action {
            Action::Finish => {
                finished = true;
                break;
            }
            Action::ForceIds(ids, text) => {
                if let Some(m) = &mut machine {
                    m.accept_forced(&ids, &text);
                }
                total_len += ids.len();
                queue.extend_from_slice(&ids);
            }
            Action::Decide(allowed) => {
                if !queue.is_empty() {
                    let (l, p) = forward(
                        &mut loaded.pool[0],
                        &io,
                        &queue,
                        total_len,
                        std::mem::take(&mut past),
                    )?;
                    logits = l;
                    past = p;
                    queue.clear();
                }
                let pick = if allowed.is_empty() {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .ok_or_else(|| ("inference_err".into(), "empty logits".into()))?
                } else {
                    argmax_masked(&logits, &allowed)
                        .ok_or_else(|| ("constraint_err".into(), "empty allowed set".into()))?
                };
                n_generated += 1;
                let picked_text = tc.decoded(pick).to_string();
                log::debug!(
                    "step {n_generated}: pick {pick} {:?} kv_layers={}",
                    picked_text,
                    past.len()
                );
                let stopped = stop_ids.contains(&pick)
                    || stop_strings
                        .iter()
                        .any(|s| picked_text.contains(s.as_str()));
                // the pick is context for the next forward in every path —
                // without it the next step would reuse stale logits
                total_len += 1;
                queue.push(pick);
                match &mut machine {
                    Some(m) => {
                        let feed = m.consume(pick, &tc, &tokenizer)?;
                        if !feed.is_empty() {
                            total_len += feed.len();
                            queue.extend_from_slice(&feed);
                        }
                    }
                    None => {
                        // the stop token itself is not included in the output
                        if !stopped {
                            free_out.push_str(&picked_text);
                        }
                    }
                }
                if stopped {
                    finished = true;
                    break;
                }
                if n_generated >= max_new_tokens {
                    break;
                }
            }
        }
    }

    match machine {
        Some(m) => Ok(GenOutput {
            text: m.text().to_string(),
            n_new_tokens: n_generated,
            finished,
        }),
        None => Ok(GenOutput {
            text: free_out,
            n_new_tokens: n_generated,
            finished,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FillMachine unit tests: no model needed; a tiny WordPiece
    // tokenizer gives deterministic ids and TokenClass::build derives the
    // mask sets from the same vocab, so masks and ids always agree. ----

    fn test_tokenizer() -> (TokenizerHandle, std::path::PathBuf) {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tid = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("onnxwrap_gen_test_{tid}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 200,
                "vocab": {
                    "[UNK]": 0, "\"": 1, "##\"": 2, ":": 3, "##:": 4, "0": 5,
                    ",\"": 6, ",": 7, "}": 8, "]": 9, "true": 10, "false": 11,
                    "##years": 12, "##_": 13, "##java": 14, "##band": 15,
                    "##led": 16, "##stacks": 17, "[": 18,
                    "jun": 19, "##ior": 20, "sen": 21
                }
            },
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
        });
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&vocab).unwrap()).unwrap();
        let tok = TokenizerHandle::from_file(&path).unwrap();
        (tok, path)
    }

    fn schema() -> Value {
        serde_json::json!({
            "base": "v1",
            "fields": [
                {"name": "years_java", "type": "numeric", "required": true},
                {"name": "band", "type": "string", "enum": ["junior", "senior"]},
                {"name": "led", "type": "bool"},
                {"name": "stacks", "type": "list_string"}
            ]
        })
    }

    #[test]
    fn schema_parse_rejects_unknown_type() {
        let s = serde_json::json!({"fields": [{"name": "x", "type": "float"}]});
        assert!(FillMachine::from_schema_json(&s).is_err());
        assert!(FillMachine::from_schema_json(&schema()).is_ok());
    }

    #[test]
    fn numeric_terminal_sep_quote_advances_with_sep_emitted() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&schema()).unwrap();
        let frag = "\"years_java\": ";
        let first = tok.encode(frag, false).unwrap().ids[0];
        let feed = m.consume(first, &tc, &tok).unwrap();
        assert_eq!(m.state, State::Numeric { idx: 0, digits: 0 });
        assert_eq!(m.text(), format!("{{\"years_java\": "));
        assert_eq!(feed, tok.encode(&frag[1..], false).unwrap().ids);
        let digit = tc.digit[0];
        m.consume(digit, &tc, &tok).unwrap();
        assert_eq!(m.state, State::Numeric { idx: 0, digits: 1 });
        let sep = tc.sep_quote[0];
        m.consume(sep, &tc, &tok).unwrap();
        assert_eq!(
            m.state,
            State::FieldBoundary {
                next: 1,
                sep_emitted: true
            }
        );
        assert!(m.text().ends_with(",\""));
    }

    #[test]
    fn numeric_needs_one_digit_before_terminal() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&schema()).unwrap();
        let first = tok.encode("\"years_java\": ", false).unwrap().ids[0];
        m.consume(first, &tc, &tok).unwrap();
        match m.next_action(&tc, &tok).unwrap() {
            Action::Decide(allowed) => {
                assert_eq!(allowed, tc.digit, "first numeric step admits digits only");
            }
            a => panic!("expected Decide, got {a:?}"),
        }
    }

    #[test]
    fn numeric_terminal_excludes_sep_quote_after_last_field() {
        let s = serde_json::json!({"fields": [
            {"name": "a", "type": "numeric", "required": true}
        ]});
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&s).unwrap();
        let first = tok.encode("\"a\": ", false).unwrap().ids[0];
        m.consume(first, &tc, &tok).unwrap();
        let digit = tc.digit[0];
        m.consume(digit, &tc, &tok).unwrap();
        match m.next_action(&tc, &tok).unwrap() {
            Action::Decide(allowed) => {
                assert!(!allowed.iter().any(|i| tc.sep_quote.contains(i)));
                assert!(allowed.iter().any(|i| tc.brace.contains(i)));
            }
            a => panic!("expected Decide, got {a:?}"),
        }
    }

    #[test]
    fn required_fields_gate_the_object_close() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let m = FillMachine::from_schema_json(&schema()).unwrap();
        match m.next_action(&tc, &tok).unwrap() {
            Action::Decide(allowed) => {
                assert!(!allowed.iter().any(|i| tc.brace.contains(i)));
            }
            a => panic!("expected Decide, got {a:?}"),
        }
        let s = serde_json::json!({"fields": [
            {"name": "a", "type": "numeric"},
            {"name": "b", "type": "bool", "required": true}
        ]});
        let mut m2 = FillMachine::from_schema_json(&s).unwrap();
        let first = tok.encode("\"a\": ", false).unwrap().ids[0];
        m2.consume(first, &tc, &tok).unwrap();
        match m2.next_action(&tc, &tok).unwrap() {
            Action::Decide(allowed) => {
                assert!(!allowed.iter().any(|i| tc.brace.contains(i)));
            }
            a => panic!("expected Decide, got {a:?}"),
        }
    }

    #[test]
    fn literal_branch_and_completion() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&schema()).unwrap();
        let quote = tc.quote[0];
        // field 0 (numeric): fragment + one digit + `,"`
        m.consume(quote, &tc, &tok).unwrap();
        m.consume(tc.digit[0], &tc, &tok).unwrap();
        m.consume(tc.sep_quote[0], &tc, &tok).unwrap();
        // field 1 (string enum): fragment re-emitted with `band": `; the
        // literal walk runs over `junior"` vs `senior"` token spellings
        m.consume(quote, &tc, &tok).unwrap();
        assert!(matches!(
            m.state,
            State::Literal {
                idx: 1,
                depth: 0,
                ..
            }
        ));
        let jun = tok.encode("junior\"", false).unwrap().ids[0];
        m.consume(jun, &tc, &tok).unwrap();
        assert!(matches!(
            m.state,
            State::Literal {
                idx: 1,
                depth: 1,
                ..
            }
        ));
        // the surviving single candidate is force-fed to completion
        for _ in 0..4 {
            match m.next_action(&tc, &tok).unwrap() {
                Action::ForceIds(ids, text) => m.accept_forced(&ids, &text),
                _ => break,
            }
        }
        assert_eq!(
            m.state,
            State::FieldBoundary {
                next: 2,
                sep_emitted: false
            }
        );
        assert!(m.text().ends_with("junior\""));
        // field 2 (bool): one pick resolves the whole literal
        m.consume(quote, &tc, &tok).unwrap();
        assert!(matches!(
            m.state,
            State::Literal {
                idx: 2,
                depth: 0,
                ..
            }
        ));
        let true_id = tok.encode("true", false).unwrap().ids[0];
        m.consume(true_id, &tc, &tok).unwrap();
        assert_eq!(
            m.state,
            State::FieldBoundary {
                next: 3,
                sep_emitted: false
            }
        );
        assert!(m.text().ends_with("true"));
    }

    #[test]
    fn list_flow_forces_brackets_and_elements() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&schema()).unwrap();
        let quote = tc.quote[0];
        // run to field 3 (list_string)
        m.consume(quote, &tc, &tok).unwrap(); // field 0 fragment
        m.consume(tc.digit[0], &tc, &tok).unwrap();
        m.consume(tc.sep_quote[0], &tc, &tok).unwrap();
        m.consume(quote, &tc, &tok).unwrap(); // field 1 fragment (enum literal)
        let jun = tok.encode("junior\"", false).unwrap().ids[0];
        m.consume(jun, &tc, &tok).unwrap();
        for _ in 0..4 {
            match m.next_action(&tc, &tok).unwrap() {
                Action::ForceIds(ids, text) => m.accept_forced(&ids, &text),
                _ => break,
            }
        }
        assert_eq!(
            m.state,
            State::FieldBoundary {
                next: 2,
                sep_emitted: false
            }
        );
        m.consume(quote, &tc, &tok).unwrap(); // field 2 fragment (bool)
        m.consume(tok.encode("true", false).unwrap().ids[0], &tc, &tok)
            .unwrap();
        m.consume(quote, &tc, &tok).unwrap(); // field 3 fragment
        assert_eq!(m.state, State::ListOpen { idx: 3 });
        // machine forces `[`
        match m.next_action(&tc, &tok).unwrap() {
            Action::ForceIds(ids, text) => m.accept_forced(&ids, &text),
            a => panic!("expected ForceIds, got {a:?}"),
        }
        assert!(m.text().ends_with("\"stacks\": ["));
        assert_eq!(m.state, State::ListFirst { idx: 3 });
        // element 1: quote, content, quote → boundary; comma → after-comma
        m.consume(quote, &tc, &tok).unwrap();
        assert_eq!(m.state, State::ListElem { idx: 3, items: 0 });
        m.consume(tok.encode("true", false).unwrap().ids[0], &tc, &tok)
            .unwrap();
        m.consume(quote, &tc, &tok).unwrap();
        assert_eq!(m.state, State::ListBoundary { idx: 3, items: 1 });
        m.consume(tc.comma[0], &tc, &tok).unwrap();
        assert_eq!(m.state, State::ListAfterComma { idx: 3, items: 1 });
        match m.next_action(&tc, &tok).unwrap() {
            Action::ForceIds(ids, text) => {
                assert_eq!(text, " \"");
                m.accept_forced(&ids, &text);
            }
            a => panic!("expected ForceIds, got {a:?}"),
        }
        assert_eq!(m.state, State::ListElem { idx: 3, items: 1 });
        // element 2 then `]` closes the list (the forced ` "` already
        // opened it)
        m.consume(tok.encode("false", false).unwrap().ids[0], &tc, &tok)
            .unwrap();
        m.consume(quote, &tc, &tok).unwrap();
        m.consume(tc.bracket[0], &tc, &tok).unwrap();
        assert_eq!(
            m.state,
            State::FieldBoundary {
                next: 4,
                sep_emitted: false
            }
        );
        // after the last field the machine forces `}`
        match m.next_action(&tc, &tok).unwrap() {
            Action::ForceIds(ids, text) => m.accept_forced(&ids, &text),
            a => panic!("expected ForceIds, got {a:?}"),
        }
        assert_eq!(m.state, State::Done);
        assert!(m.text().ends_with("[\"true\", \"false\"]}"));
        let parsed: Value = serde_json::from_str(m.text()).expect("machine output is valid json");
        assert_eq!(parsed["stacks"], serde_json::json!(["true", "false"]));
    }

    #[test]
    fn full_greedy_drive_produces_valid_json() {
        let (tok, _p) = test_tokenizer();
        let tc = TokenClass::build(&tok);
        let mut m = FillMachine::from_schema_json(&schema()).unwrap();
        for _ in 0..200 {
            match m.next_action(&tc, &tok).unwrap() {
                Action::Finish => break,
                Action::ForceIds(ids, text) => m.accept_forced(&ids, &text),
                Action::Decide(allowed) => {
                    let pick = *allowed.last().unwrap();
                    m.consume(pick, &tc, &tok).unwrap();
                }
            }
        }
        let parsed: Value = serde_json::from_str(m.text()).expect("valid json");
        assert_eq!(parsed["years_java"], serde_json::json!(0));
    }
}
