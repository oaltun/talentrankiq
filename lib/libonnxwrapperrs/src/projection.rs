//! ColBERT projection chain applied to per-token transformer states.
//!
//! PyLate `Dense` modules (lightonai/pylate, models/Dense.py) ship as
//! safetensors with `linear.weight` [out, in] and — when the dimensions
//! change — a learned `residual.weight` [out, in]. The forward pass is
//! `h' = linear(h) + residual(h)`, and pylate's `encode` L2-normalizes the
//! token vectors (`normalize_embeddings=True` by default). mLateOn chains
//! three such modules: 768 → 1536 → 768 → 128 (see
//! bench/bench1/design/17-late-interaction.md).
//!
//! The safetensors parsing here is deliberately minimal: f32 tensors only,
//! header + JSON + data, no compression.

use std::collections::HashMap;
use std::path::PathBuf;

/// One Dense step: `out = w @ h (+ residual @ h)`.
#[derive(Debug)]
struct Step {
    w: Vec<f32>,
    residual: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

/// A chain of Dense steps ending in the projection dim.
#[derive(Debug)]
pub struct ProjectionChain {
    steps: Vec<Step>,
    pub in_dim: usize,
    pub out_dim: usize,
}

type LoadErr = (String, String);

impl ProjectionChain {
    /// Load and validate a chain of safetensors Dense files.
    ///
    /// `expected_in_dim` is the transformer hidden size; the chain must start
    /// there and every step's input must match the previous step's output.
    /// File order in `files` is the application order.
    pub fn load(files: &[PathBuf], expected_in_dim: usize) -> Result<Self, LoadErr> {
        if files.is_empty() {
            return Err((
                "projection_err".into(),
                "no projection files given (opts.projection_files is empty)".into(),
            ));
        }

        let mut steps = Vec::with_capacity(files.len());
        let mut cur_in = expected_in_dim;
        for file in files {
            let st = SafeTensorsFile::parse(std::fs::read(file).map_err(|e| {
                (
                    "projection_err".into(),
                    format!("failed to read {}: {e}", file.display()),
                )
            })?)
            .map_err(|e| ("projection_err".into(), format!("{}: {e}", file.display())))?;

            let (shape, w) = st
                .tensor("linear.weight")
                .map_err(|e| ("projection_err".into(), format!("{}: {e}", file.display())))?;
            let [out_dim, in_dim] = shape.as_slice() else {
                return Err((
                    "projection_err".into(),
                    format!(
                        "{}: linear.weight must be 2D [out, in], got {shape:?}",
                        file.display()
                    ),
                ));
            };
            let residual = match st.tensor("residual.weight") {
                Ok((rshape, r)) if rshape == shape => Some(r),
                Ok((rshape, _)) => {
                    return Err((
                        "projection_err".into(),
                        format!(
                            "{}: residual.weight shape {rshape:?} != linear.weight {shape:?}",
                            file.display()
                        ),
                    ));
                }
                Err(_) => None,
            };

            if *in_dim != cur_in {
                return Err((
                    "projection_err".into(),
                    format!(
                        "{}: expects input dim {in_dim}, chain is at {cur_in}",
                        file.display()
                    ),
                ));
            }
            steps.push(Step {
                w,
                residual,
                in_dim: *in_dim,
                out_dim: *out_dim,
            });
            cur_in = *out_dim;
        }

        let in_dim = steps[0].in_dim;
        let out_dim = steps[steps.len() - 1].out_dim;
        Ok(Self {
            steps,
            in_dim,
            out_dim,
        })
    }

    /// Apply the chain to a single token vector, then L2-normalize.
    pub fn project_token(&self, token: &[f32]) -> Vec<f32> {
        let mut cur = token.to_vec();
        for step in &self.steps {
            let mut out = matvec_add(&step.w, &cur, step.in_dim, step.out_dim);
            if let Some(res) = &step.residual {
                let r = matvec_add(res, &cur, step.in_dim, step.out_dim);
                for (o, &x) in out.iter_mut().zip(&r) {
                    *o += x;
                }
            }
            cur = out;
        }
        l2_normalize(&mut cur);
        cur
    }
}

fn matvec_add(m: &[f32], v: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for (r, row) in m.chunks_exact(in_dim).enumerate() {
        let mut s = 0.0f32;
        for k in 0..in_dim {
            s += row[k] * v[k];
        }
        out[r] = s;
    }
    out
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

struct SafeTensorsFile {
    data: Vec<u8>,
    entries: HashMap<String, (Vec<usize>, usize, usize)>,
}

impl SafeTensorsFile {
    fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err(format!(
                "file too small for safetensors ({} bytes)",
                bytes.len()
            ));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_start = 8 + header_len;
        if data_start > bytes.len() {
            return Err(format!("header length {header_len} exceeds file size"));
        }
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..data_start])
            .map_err(|e| format!("invalid safetensors header JSON: {e}"))?;
        let obj = header
            .as_object()
            .ok_or_else(|| "safetensors header is not an object".to_string())?;

        let mut entries = HashMap::new();
        for (name, spec) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = spec
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("tensor {name}: missing dtype"))?;
            if dtype != "F32" {
                return Err(format!(
                    "tensor {name}: dtype {dtype} not supported (F32 only)"
                ));
            }
            let shape: Vec<usize> = spec
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
                .ok_or_else(|| format!("tensor {name}: missing shape"))?;
            let offsets: Vec<usize> = spec
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
                .ok_or_else(|| format!("tensor {name}: missing data_offsets"))?;
            if offsets.len() != 2 || offsets[1] < offsets[0] {
                return Err(format!("tensor {name}: bad data_offsets"));
            }
            let (begin, end) = (offsets[0], offsets[1]);
            let nelem: usize = shape.iter().product();
            if end - begin != nelem * 4 {
                return Err(format!(
                    "tensor {name}: data range {} bytes != {} elements * 4",
                    end - begin,
                    nelem
                ));
            }
            entries.insert(name.clone(), (shape, data_start + begin, data_start + end));
        }

        Ok(Self {
            data: bytes,
            entries,
        })
    }

    fn tensor(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let (shape, begin, end) = self
            .entries
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found"))?;
        let mut out = Vec::with_capacity((end - begin) / 4);
        for chunk in self.data[*begin..*end].chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok((shape.clone(), out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Build minimal safetensors bytes for f32 tensors (row-major).
    fn make_safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape, vals) in tensors {
            let begin = data.len();
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [begin, data.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    fn write_st(dir: &Path, name: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, make_safetensors(tensors)).unwrap();
        path
    }

    #[test]
    fn parses_safetensors() {
        let dir = std::env::temp_dir().join(format!("onnxproj_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_st(
            &dir,
            "a.safetensors",
            &[(
                "linear.weight",
                vec![2, 3],
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )],
        );
        let st = SafeTensorsFile::parse(std::fs::read(&path).unwrap()).unwrap();
        let (shape, vals) = st.tensor("linear.weight").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(st.tensor("missing").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chain_applies_residual_math() {
        let dir = std::env::temp_dir().join(format!("onnxproj_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // step 1: in 2 -> out 2 with residual = 0.5 * identity
        let f1 = write_st(
            &dir,
            "1_Dense.safetensors",
            &[
                ("linear.weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                ("residual.weight", vec![2, 2], vec![0.5, 0.0, 0.0, 0.5]),
            ],
        );
        // step 2: in 2 -> out 1, no residual
        let f2 = write_st(
            &dir,
            "3_Dense.safetensors",
            &[("linear.weight", vec![1, 2], vec![1.0, 1.0])],
        );
        let chain = ProjectionChain::load(&[f1, f2], 2).unwrap();
        assert_eq!(chain.in_dim, 2);
        assert_eq!(chain.out_dim, 1);

        // h = [3, 4]: step1 = h + 0.5h = [4.5, 6]; step2 = 4.5 + 6 = 10.5 → normalized = 1.0
        let out = chain.project_token(&[3.0, 4.0]);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chain_rejects_dim_mismatch() {
        let dir = std::env::temp_dir().join(format!("onnxproj_test3_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = write_st(
            &dir,
            "a.safetensors",
            &[("linear.weight", vec![2, 5], vec![0.0; 10])],
        );
        let err = ProjectionChain::load(&[f], 3).unwrap_err();
        assert_eq!(err.0, "projection_err");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn project_token_normalizes() {
        let dir = std::env::temp_dir().join(format!("onnxproj_test4_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = write_st(
            &dir,
            "a.safetensors",
            &[("linear.weight", vec![1, 2], vec![3.0, 4.0])],
        );
        let chain = ProjectionChain::load(&[f], 2).unwrap();
        let out = chain.project_token(&[1.0, 1.0]);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Golden test against real mLateOn weights. Skipped unless
    /// `MLATEON_WEIGHTS_DIR` (default: /tmp/kilo/mlateon_weights) contains
    /// `{1,2,3}_Dense.safetensors`. The expected values in
    /// tests/fixtures/mlateon_projection_golden.json were computed with numpy
    /// from the same weights: h = chain(x) with per-token L2 normalization,
    /// x = default_rng(42).standard_normal((16, 768)).
    #[test]
    fn mlateon_projection_matches_numpy() {
        const STEPS: [&str; 3] = [
            "1_Dense.safetensors",
            "2_Dense.safetensors",
            "3_Dense.safetensors",
        ];
        let dir = std::env::var("MLATEON_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/tmp/kilo/mlateon_weights".to_string());
        let files: Vec<PathBuf> = STEPS
            .iter()
            .map(|s| std::path::Path::new(&dir).join(s))
            .collect();
        if files.iter().any(|f| !f.exists()) {
            eprintln!("skipping: weights not found in {dir}");
            return;
        }

        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/mlateon_projection_golden.json"
            ))
            .unwrap(),
        )
        .unwrap();

        let chain = ProjectionChain::load(&files, 768).unwrap();
        assert_eq!(chain.in_dim, 768);
        assert_eq!(chain.out_dim, 128);

        let parse = |key: &str| -> Vec<Vec<f64>> {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect()
                })
                .collect()
        };
        let input = parse("input");
        let expected = parse("expected");
        assert_eq!(input.len(), 16);
        assert_eq!(expected.len(), 16);

        let mut worst = 0.0f64;
        for (token_in, token_expected) in input.iter().zip(expected.iter()) {
            let out = chain.project_token(&token_in.iter().map(|&v| v as f32).collect::<Vec<_>>());
            assert_eq!(out.len(), 128);
            for (got, want) in out.iter().zip(token_expected.iter()) {
                worst = worst.max((*got - *want as f32).abs() as f64);
            }
            let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "output not unit-norm: {norm}");
        }
        assert!(worst < 1e-3, "max deviation from numpy golden: {worst}");
        eprintln!("max deviation from numpy golden: {worst}");
    }
}
