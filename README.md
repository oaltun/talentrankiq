# TalentRankIQ

Local-first desktop app for ranking applicant CVs against job advertisements.
Everything runs on-device (Tauri + Rust + React); no CV data leaves the machine.

![platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue)
![rust](https://img.shields.io/badge/rust-%2B%20libs-orange)
![react](https://img.shields.io/badge/react-gray)

## What it does

- Import a batch of CVs (PDF/DOCX/TXT) into a job bundle and parse them locally.
- Write a job ad + configurable rubric (rankable criteria); the pipeline turns
  both into per-space rubric vectors.
- Watch the CV directory for new files and re-rank incrementally — no re-embed
  of unchanged input.
- Rank the whole corpus with a **progressive cascade**: a cheap multilingual
  "scout" pass orders everyone, then an expensive on-device embedder (MLX in macOS)
  and then an optional in-house developed model (Judge) refine the top of the sheet wave by wave.
- Show per-candidate scores, rank, evidence and reasoning; export results to
  CSV; record candidate feedback ("like") that feeds back into scoring.

## Ranking model

TalentRankIQ ranks candidates in two layers:

1. **Embedder ranking** — always available. Fast on-device embedding + rubric cascade ranking. 

The cascade guarantees convergence (a run ends when every candidate holds heavy-space chunks and every pool member holds a judge grade) and monotone improvement (a wave only ever replaces grades with better ones), so the top of the sheet stabilises early. 

System-capacity budgeting (`libsyscap`) scales wave size to the machine, and a `holding` reason code surfaces budget-limited rows instead of fake results.

2. **Judge** — the pairwise in-house AI that resolves close candidates in the frozen top-P pool. Highly recommended!

The judge is a separately distributed **judge artifact** and is **licensed**: buy a judge license, place the signed key file in your user config dir, and the judge is enabled. 

Without a license the full app still works with embedder ranking.

Judge artifacts are **not** part of this repository. Artifacts are purchased and delivered through [talentrankiq.com](https://talentrankiq.com).

> **Licensing note (Elastic License 2.0, ELv2).** This code is *source-available*,
> not open source. You may use, copy, modify, and redistribute it, but you may
> **not** (a) offer it to third parties as a hosted or managed service, or
> (b) circumvent the license-key functionality that enables judge functionality
> (see `lib/libtriqui-license`).

## Repository layout

- `gui/` — Tauri app (React frontend in `gui/src`, Rust bridge in `gui/src-tauri`)
- `lib/` — Rust workspace crates:
  - `libtriqui-core` — domain: SQLite store, cascade ranking, CV import, scoring
  - `libtriqui-cli` — headless CLI (`triq`): serve loop over stdio JSONL, plus tools
  - `libtriqui-client` — gui↔engine wire contract (JSONL protocol, DTOs, client)
  - `libtriqui-license` — judge entitlement (Ed25519-verified license files)
  - `libmlxembed` / `libonnxwrapperrs` / `libtokenizer` — on-device inference
  - `libmodelcaller` / `libmodelcallerclient` / `libmodeldownloader` /
    `libmodelstorage` — model serving & management
  - `libcvparser` / `libcvevidence` — CV parsing and evidence extraction
  - `libjudge` — judge artifact schema, feature assembly, judge verdicts
  - `libsyscap` — system-capacity budgeting for cascade waves
  - `libconfigreader` — config helper
- `vendor/` — vendored libraries

## Build & run

Prerequisites: Rust (stable), Node ≥ 20, a C++ toolchain (CMake/MSVC/Clang for the vendored libs), and the platform toolchain for Tauri.

Build order: compile the Rust dependencies first (`lib/` crates plus vendored `vendor/mlxcel` via `libmlxembed`), then build/run the GUI.

### 1. Compile all Rust dependencies

At the repo root:

```sh
cargo build --release
```

Builds the whole workspace — every `lib/` crate and `gui/src-tauri` — including `vendor/mlxcel`, which is linked through `libmlxembed`. Use `cargo build` (dev profile) for faster iteration.

### 2. Build & run the GUI

```sh
cd gui
npm install
npm run tauri:dev
```

`npm run tauri:dev` rebuilds Rust in dev mode and launches the app. For a release bundle:

```sh
npm run tauri:build
```

## License

Licensed under the [Elastic License 2.0 (ELv2)](LICENSE). See the [license
note above](#ranking-model) for the source-available restrictions. Judge
artifacts are separate commercial products and not covered by this license.
