# Tauri CLI Bridge Fix Plan

## Root Cause

The new GUI's Tauri shell (`src-tauri/src/lib.rs`) only registers `greet` as a Tauri command. All other commands (`download_embedder`, `list_embedders`, etc.) are missing from the `invoke_handler`, so `appInvoke` fails silently in Tauri mode while the Playwright e2e tests pass (using the Vite proxy that forwards to the real `triq serve` CLI).

## Design Principles

- **Exact e2e parity**: The Tauri app must behave identically to the Playwright e2e test (`triq serve` subprocess).
- **Minimal CLI edit**: Only one CLI edit needed (add `OUT_DIR` to build script).
- **Single dispatcher**: One Tauri command (`triq_dispatch`) that forwards to the CLI client.
- **No per-command registration**: Avoids boilerplate and matches e2e test structure.

## Decision

Use the **subprocess bridge** approach (same as old GUI) with `libtriqui-client` as a path dependency. This gives:
- Exact e2e parity (same `triq serve` process, same env, same DB).
- Minimal CLI edit (only `build.rs` modification).
- No heavy in-process dependencies.
- Matches the e2e test path exactly.

## Tasks

### 1. Add `libtriqui-client` path dependency
- `src-tauri/Cargo.toml`: add `libtriqui-client = { path = "../../talentrankiq_cli/lib/libcli-client" }`
- Verify `libcli-client/build.rs` writes `protocol-schema.json` and `protocol.ts` to `OUT_DIR` (add `OUT_DIR` path if needed).

### 2. Wire `libtriqui-client` in `src-tauri/lib.rs`
- `mod cli_subprocess;`
- `mod commands;`
- `mod triqi_client;`
- `mod utils;`
- Ensure `lib.rs` imports `triqi_client::TriqiClient` and `triqi_client::ClientError`.

### 2. Add `triq_dispatch` Tauri command
- In `src-tauri/src/lib.rs`:
  - Add a new command `triq_dispatch` that:
    - Takes `cmd: String` and `params: serde_json::Value`.
    - Calls `client.call(&cmd, &params)`.
    - Returns the parsed result.
- Register it in `tauri::generate_handler![]` with all commands the frontend calls:
  - `list_embedders`, `download_embedder`, `remove_embedder`, `set_embedding_model`, `make_default_embedder`, `download_status`, `download_complete`, `get_scout_model`, `get_comparator_artifact`, `load_comparator_artifact`, `remove_comparator_artifact`, `list_job_summaries`, `create_job`, `archive_job`, `delete_job`, `watch_job`, `get_job_detail`, `get_job_ad`, `set_job_ad_path`, `reload_job_ad`, `import_cv_directory`, `convert_pending`, `scout_pending`, `heavy_pending`, `set_job_cv_dir`, `update_job`, `update_job_weights`, `add_criterion`, `update_criterion`, `delete_criterion`, `empty_candidates`, `export_results_csv`, `export_job_bundle`, `list_candidates`, `get_candidate`, `candidate_neighbors`, `apply_feedback`, `reset_feedback`, `list_liked_candidate_ids`, `get_job_ui_state`, `get_rank_state`, `start_rank`, `cancel_rank`, `get_sort_report`, `runtime_status`, `start_runtime`, `system_budget`, `chunk_match`, `handshake`, `greet`.

### 3. Capabilities and dev command

- `src-tauri/capabilities/default.json`: add `"dialog:default"` (required for `open`/`save`).
- `src-tauri/tauri.conf.json`: 
  - `beforeDevCommand`: `"(cd ../talentrankiq_cli && cargo build) && bun run dev"` (ensures `triq` binary exists).
  - `build.removeUnusedCommands: false` (to preserve all Tauri commands).
  - `productName`, `version`, `ident` already set correctly.

### 6. Frontend: unify invoke and fix param names
- `src/lib/invoke.ts`: add `appInvoke` (already exists) and ensure `isTauri()` uses it.
- Replace all raw `invoke` calls with `appInvoke` in:
  - `src/services/rank.ts`
  - `src/services/feedback.ts`
  - `src/services/criteria.ts`
  - `src/services/results-csv.ts`
  - `src/services/job-bundle.ts`
  - `src/services/match-lens.ts`
- Ensure all param objects use camelCase (e.g., `job_ad_text` → `jobAdText`).
- Verify `listenEvent` in `src/lib/events.ts` still works for CLI events (no change needed).

### 6. Surface Tauri bridge failures

- In `src/services/models.ts`, modify `listEmbedders` and `getScoutModel` to only fallback to `BROWSER_*` when `!isTauri()`:
  ```ts
  if (!isTauri() && error.message.includes("not found")) {
    // fall back to BROWSER_EMBEDDERS
  } else {
    throw error; // let Tauri show the real error
  }
  ```
  This ensures Tauri users see real errors instead of fake "no models" state.

### 7. Test hardening (recommended)

- Add a manual check: `bun run tauri dev` → open Models page → remove all models → download quick model → verify Remove button appears.
- Add `tauri-pilot` e2e test: copy old GUI's `scenarios/app-shell.toml` and add a step that downloads a model. Mark as recommended to prevent future regressions.

## Risks & Mitigations

- **`build.removeUnusedCommands`**: Keep it `true`; it only strips plugin commands, not app commands.
- **`TRIQI_MODEL_DIR`**: Do NOT set it — use default cache dir (`~/.cache/talentrankiq/models`) so app and e2e share models.
- **`TRIQ_APP_DIR`**: Set to `app.path().app_data_dir()?.join("triq")` to match the Vite proxy’s `<support>/<id>/triq` layout.
- **CLI build script**: Add `OUT_DIR` path to `libcli-client/build.rs` so it writes `protocol-schema.json` and `protocol.ts` to `OUT_DIR` (e.g., `OUT_DIR="/tmp/triq_build"`).
- **Cargo.toml**: Path dependency is safe; CLI build script is idempotent.

## Validation

1. `bun run typecheck` (no errors).
2. `bun run lint` (exit 0).
3. `bun run build` (front + Tauri).
4. `bun run tauri dev`:
   - Open Models page.
   - Remove all models (click "Remove all" or use debug panel).
   - Click "Download Models" → "Download quick model".
   - Verify progress bar updates and "Remove" button appears.
   - Verify files exist in `~/.cache/talentrankiq/models/<slug>`.
   - Verify `list_embedders` returns downloaded models (not `BROWSER_EMBEDDERS`).
5. (Optional) Run `tauri-pilot` scenario: verify the app opens, navigates to Models, downloads a model, and the Remove button appears.

## Open Questions

- Does `libtriqui-client`’s `call` method return errors that `appInvoke` propagates correctly? yes. (Test with a failing command like `list_non_existent`.)
- Does the `OUT_DIR` build script need to be part of CI? (Yes — ensure CI runs `bun run tauri dev` without breaking.)
- Should `TRIQ_APP_DIR` be set explicitly in `tauri.conf.json` or rely on `app_data_dir().join("triq")`? (Use `app_data_dir().join("triq")`.)
