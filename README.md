# TalentRankIQ

Local-first desktop app for ranking applicant CVs against a job advertisement.
Everything runs on-device (Tauri + Rust + React); no CV data leaves the machine.

![platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-blue)
![rust](https://img.shields.io/badge/rust-+-libs-orange)
![react](https://img.shields.io/badge/react-19-blue)

## What it does

- Import a batch of CVs (PDF/DOCX) into a job bundle and parse them locally.
- Write a job ad + configurable rubric (rankable criteria); the pipeline turns
  both into per-space rubric vectors.
- Watch the CV directory for new files and re-rank incrementally — no re-embed
  of unchanged input.
- Rank the whole corpus with a **progressive cascade**: a cheap multilingual
  "scout" pass orders everyone, then an expensive on-device embedder (MLX)
  and an optional LLM judge refine the top of the sheet wave by wave.
- Show per-candidate scores, rank, evidence and reasoning; export results to
  CSV; record candidate feedback ("like") that feeds back into scoring.

## Ranking model

TalentRankIQ ranks candidates in two layers:

1. **Embedder ranking** — always available. Fast on-device embedding + rubric
   cascade ranking:

   ```text
   scout.run()                      # whole corpus, once — every candidate gets a total
   ensure_vectors(heavy)            # the rubric in the heavy space
   pool = freeze_pool(top P)        # frozen: judge win-rates are set-relative
   while not heavy.has_all(job) or not judge.has_all(job):
       slice = heavy.next_slice(k1) # best-current-order first
       heavy.embed(slice)           # → stage_scores(heavy)
       judge.run(pool)              # → candidate_scores[ad_judge], memoized per pair
       rescore(slice)               # → points + rank + display rows
   ```

   The cascade guarantees convergence (a run ends when every candidate holds
   heavy-space chunks and every pool member holds a judge grade) and monotone
   improvement (a wave only ever replaces grades with better ones), so the top
   of the sheet stabilises early. System-capacity budgeting (`libsyscap`)
   scales wave size to the machine, and a `holding` reason code surfaces
   budget-limited rows instead of fake results.

2. **Judge** — the pairwise LLM judge that resolves close candidates in the
   frozen top-P pool. The judge is a separately distributed **judge artifact**
   and is **licensed**: buy a judge license, place the signed key file in your
   user config dir, and the judge is enabled. Without a license the full app
   still works with embedder ranking.

Judge artifacts are **not** part of this repository. Artifacts are purchased
and delivered through [talentrankiq.com](https://talentrankiq.com).

> **Licensing note (Elastic License 2.0, ELv2).** This code is *source-available*,
> not open source. You may use, copy, modify, and redistribute it, but you may
> **not** (a) offer it to third parties as a hosted or managed service, or
> (b) circumvent the license-key functionality that enables judge functionality
> (see `lib/libtriqui-license`).

## Architecture

Thin-GUI + headless serve loop ([`gui/design/app1/thin-gui.md`](gui/design/app1/thin-gui.md)):

- **GUI** (`gui/`) — React UI that only owns presentation. It talks to the
  engine through `TriqiClient` (`libtriqui-client`), a shared wire contract
  (envelope, request params, DTOs) so both sides cannot drift.
- **Serve loop** (`libtriqui-cli`) — the long-lived headless engine: DB,
  preferences, `ikapp.yaml`, engine runtime, model management, and CV-dir
  watching all live here. In the app it runs **in-process** on a thread
  (`InProcessServe`) — one binary, no child process, no RPC over IPC edges
  the GUI needs.
- **Core** (`libtriqui-core`) — the domain, Tauri-free: SQLite store, CV
  import, rubric scoring, embedding backends, the cascade, resort and feedback.
- **Model stack** — `libmlxembed` (on-device embedder, e.g. Qwen3/Gemma3
  pooling), `libonnxwrapperrs` (ONNX sessions + probing), `libmodelcaller`
  (+ `libmodelcallerclient` contract), `libmodeldownloader`/`libmodelstorage`
  (Hugging Face downloads, cache), `libtokenizer`, `libcvparser`,
  `libcvevidence`.

### Judge licensing

A signed JSON license file (Ed25519 signature over the license payload,
verified against a public key built into `libtriqui-license`) is placed in
`<app config dir>/triqui/license.json`. Fields include `feature`
(`judge`), `expires_at` (RFC 3339), and `signature` (base64). Verification
fails closed: missing, malformed, expired, or badly signed files simply leave
the judge disabled.

## Repository layout

- `gui/` — Tauri app (React frontend in `gui/src`, Rust bridge in `gui/src-tauri`)
- `lib/` — Rust workspace crates:
  - `libtriqui-core` — domain: SQLite store, cascade ranking, CV import, scoring
  - `libtriqui-cli` — headless engine + serve loop (`InProcessServe`)
  - `libtriqui-client` — gui↔engine wire contract
  - `libtriqui-license` — judge entitlement (Ed25519-verified license files)
  - `libmlxembed` / `libonnxwrapperrs` / `libtokenizer` — on-device inference
  - `libmodelcaller` / `libmodelcallerclient` / `libmodeldownloader` /
    `libmodelstorage` — model serving & management
  - `libcvparser` / `libcvevidence` — CV parsing and evidence extraction
  - `libjudge` — judge artifact schema, feature assembly, hand-rolled forward pass
  - `libsyscap` — system-capacity budgeting for cascade waves
  - `libconfigreader` / `libbenchresults` — config and benchmark helpers
- `vendor/` — vendored on-device inference stack

## Build & run

Prerequisites: Rust (stable), Node ≥ 20, and the platform toolchain for Tauri
(macOS: Xcode CLT + webkit2gtk on Linux).

```sh
cd gui
npm install
npm run tauri:dev
```

Useful scripts (all under `gui/`):

| Script | What it does |
|---|---|
| `npm run tauri:dev` | Run the app in dev mode |
| `npm run tauri:build` | Release build |
| `npm run tauri:run` | Debug build + run, fast iteration |
| `npm run tauri:check` | Typecheck + `tauri build --check` |
| `npm run test:all` | Vitest (frontend) + `cargo test` |
| `npm run check:all` | Full gate: typecheck, lint, format, clippy, all tests |
| `npm run fix:all` | Auto-fix lint/format/clippy |

## Development notes

- Frontend: React 19, TanStack Router/Query, Radix UI, Tailwind v4, Vite.
- Tauri command bindings are generated tests (`npm run rust:bindings`).
- `libjudge` is a contract crate: training (Burn) exports plain serde weight
  arrays; the inference path is hand-rolled in-crate with Burn-to-inference
  parity enforced by fixture vectors embedded in every artifact
  (`JudgeArtifact::verify_parity`).
- The CLI has no separate binary; hosts (the GUI, e2e contract tests) run
  `InProcessServe` in-thread. Process-global knobs such as `--cpu-only`
  emulate a slow machine for testing.

## License

Licensed under the [Elastic License 2.0 (ELv2)](LICENSE). See the [license
note above](#ranking-model) for the source-available restrictions. Judge
artifacts are separate commercial products and not covered by this license.