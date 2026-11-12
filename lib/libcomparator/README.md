# libcomparator

Plain-Rust inference for the frozen-embedder comparator (design
`bench/bench1/design/15-pairwise-comparator.md`). loads a trained comparator artifact
(JSON), rebuilds the feature assembly, and scores documents with a
hand-rolled forward pass. no burn, no ONNX — this crate is safe to
link into any production binary.

## what it owns

- **artifact contract** (`artifact.rs`) — serde + validation for the
  comparator artifact: feature spec (embedder, dim, chunking, contextual,
  variant, tau, text preprocessing), layer weights (`w[out][in]`),
  run metadata, per-block layout, and parity fixture vectors.
  `ComparatorArtifact::load` re-verifies burn-vs-hand-rolled parity
  (tol 1e-4) on every load, so a stale or corrupt artifact is a hard
  error at startup, not silently wrong scores.
- **feature assembly** (`features.rs`) — raw pair input
  `[a; e1; e2; c1; c2]` (5*dim), engineered pair input
  `[e1; e2; e1-e2; e1*e2; a; h1; h2; h1-h2; s1; s2]` (8*dim+8), and
  pointwise input `[a; e; c]` (3*dim), where `a` is the L2-normalized
  job-ad centroid, `e` the L2-normalized cv centroid, `c` the
  job-contextual cv chunk, `h = a*e`, and `s` per-cv chunk stats
  ([max, mean, top3, coverage] at tau).
- **forward pass** (`forward.rs`) — affine layers + ReLU between
  hidden layers only; pairwise models emit a comparison logit
  (`pair_prob` = sigmoid), pointwise models emit P(Good Fit) via
  softmax over 3 classes.
- **extraction** (`extract.rs`) — the `TextEmbedder` trait (batched
  `&[String] -> Vec<Vec<f64>>`), `ChunkCache` (byte-identical key and
  file format with bench step-14 `EmbedCache`:
  sha256(model + "\0" + trimmed text), first 16 bytes hex, file
  `{"embedding": [...]}`), and `FeatureExtractor` which chunks,
  dedups, batch-embeds, and assembles `DocFeatures` per cv. `ctx_size`
  chunking needs a `libtokenizer` handle; `none` does not.
- **tournament** (`tournament.rs`) — both-orders-averaged probability
  matrix, win-rate scores (default aggregator, ties = 0.5), cycle
  rate, and Bradley–Terry fallback.

## what it does NOT own

- training (see `comparator/comparatortrainer/`; burn is confined there)
- spawning modelcaller — callers provide an embedder impl of
  `TextEmbedder` over their own client, with `texts_role = None`

## layout

```
src/
├── lib.rs          # errors, l2_normalize, cosine, sigmoid
├── artifact.rs     # ComparatorArtifact serde/validate/parity/load
├── features.rs     # DocFeatures, input assembly, block layout
├── forward.rs      # ComparatorModel hand-rolled scoring
├── extract.rs      # TextEmbedder, ChunkCache, FeatureExtractor
└── tournament.rs   # probability matrix, win rates, Bradley-Terry
```

## usage sketch

```rust
let artifact = libcomparator::artifact::ComparatorArtifact::load(path)?; // parity-checked
let model = libcomparator::forward::ComparatorModel::from_artifact(&artifact)?;

let mut extractor = libcomparator::extract::FeatureExtractor::from_spec(
    &artifact.feature_spec, max_ctx, tokenizer, cache, embedder);
let docs = extractor.extract(libcomparator::extract::ObservationTexts {
    job_text, resume_texts })?;

// pairwise: win-rate scores over the tournament matrix
let p = libcomparator::tournament::pairwise_prob_matrix(&model,
    &artifact.feature_spec, &docs[0], &docs);
let scores = libcomparator::tournament::win_rate_scores(&p);

// pointwise (m0): one forward per (ad, cv), score = P(Good Fit)
let score = model.score(libcomparator::features::assemble_pointwise(
    &artifact.feature_spec, &docs[0], &docs[3]));
```

## testing

```bash
cargo test --manifest-path lib/libcomparator/Cargo.toml    # 20 unit tests
```

covers: artifact validation/parity roundtrip, feature assembly dims
and blocks, forward parity vs known weights, tournament math
(win rates, cycle rate, Bradley–Terry), cache key/format identity
with bench.

## consumers

- `comparator/comparatortrainer/` — writes artifacts (burn trains, libcomparator scores
  during validation so every metric is also a parity check)
- `bench/` — `pairwise-comparator` / `pointwise-comparator` methods link this
  crate for inference; feature prep is the same rust code end-to-end
