//! End-to-end workflows for the **gui**, driven the way the gui bridge drives
//! them: the real serve loop **in-process** (`libtriqui_cli::InProcessServe` —
//! there is no `triqui-cli` binary) over a typed
//! `libtriqui_client::TriqiClient`.
//!
//! These live here, in the gui crate's own test target, rather than in the
//! shared `tests` crate so the gui and its sibling ports (`guigpui`,
//! `guiiced`, `guigpuikit`) can each be developed and run in parallel without
//! contending for one shared binary. The split is by ownership, then by cost:
//! - **this file** — gui-facing workflows, all cheap: no engine, no models, no
//!   network. That is not just tidiness: `npm run check:all` runs `cargo test`
//!   in `src-tauri` **without** `IK_SKIP_EXPENSIVE`, so anything expensive
//!   landing here would pull real models into the gui gate.
//! - **`tests/tests/serve_contract.rs`** — gui-independent serve contracts
//!   (transport, cascade API, chunk match, syscap, recovery, engine-free bundle
//!   round trip).
//! - **`tests/tests/serve_gui_contract.rs`** — the expensive full-stack
//!   workflows. They share the on-disk model cache, so they deliberately stay
//!   in **one** binary: the lock that serialises them is an in-process
//!   `static`, which only works while they share a target.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use libtriqui_client::client::{ClientError, TriqiClient};
use libtriqui_client::params::{DownloadEmbedderParams, SetEmbeddingModelParams};
use libtriqui_client::types::DomainError;
use libtriqui_client::{EmptyParams, GetJobDetailParams, StartRankParams};
use libtriqui_cli::InProcessServe;

/// Event collector standing in for the gui's forwarder thread: drains the
/// serve event stream into a shared log the test can await on.
fn spawn_event_collector(client: &TriqiClient) -> Arc<Mutex<Vec<(String, serde_json::Value)>>> {
    let rx = client.take_event_receiver().expect("event receiver");
    let log: Arc<Mutex<Vec<(String, serde_json::Value)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&log);
    std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            sink.lock()
                .expect("event log lock")
                .push((event.event, event.data));
        }
    });
    log
}

/// create_job + update_job (the Setup pane flow) → the job id.
fn create_job_with_ad(client: &TriqiClient, title: &str, ad: &str) -> String {
    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job.id.clone(),
                title: title.into(),
                job_ad_text: Some(ad.into()),
                triqui_size: 10,
            },
        )
        .expect("set job ad");
    job.id
}

/// Same as [`expect_domain_value`], decoded into the typed [`DomainError`]
/// the frontend matches on.
fn expect_domain(
    client: &TriqiClient,
    cmd: &str,
    params: &impl serde::Serialize,
    what: &str,
) -> DomainError {
    let de: DomainError = serde_json::from_value(expect_domain_value(client, cmd, params, what))
        .unwrap_or_else(|e| panic!("{what}: not a DomainError payload: {e}"));
    de
}

/// Unwraps a failing call into the raw typed-domain payload (the gui matches
/// on the `#[serde(tag = "type")]` tag; recovery errors are their own type,
/// so this keeps them inspectable as JSON).
fn expect_domain_value(
    client: &TriqiClient,
    cmd: &str,
    params: &impl serde::Serialize,
    what: &str,
) -> serde_json::Value {
    let err = match client.call(cmd, params) {
        Err(e) => e,
        Ok(v) => panic!("{what}: expected a domain error, got ok payload {v}"),
    };
    let ClientError::Domain(pe) = err else {
        panic!("{what}: expected a domain error envelope, got {err:?}");
    };
    assert_eq!(pe.code, "domain");
    pe.domain
        .unwrap_or_else(|| panic!("{what}: no domain payload"))
}

fn fresh_app_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("gui-e2e-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Two CVs extracted into `.txt` files under `root`, imported into `job_id`,
/// and the import waited out (polls the DB for the extracted rows). Returns the
/// CV directory.
fn import_two_txt_cvs(
    client: &TriqiClient,
    root: &std::path::Path,
    app_dir: &std::path::Path,
    job_id: &str,
) -> std::path::PathBuf {
    let cv_dir = root.join(format!("cvs-{job_id}"));
    std::fs::create_dir_all(&cv_dir).unwrap();
    std::fs::write(
        cv_dir.join("alice.txt"),
        "Alice Chen\nSenior Rust engineer with 9 years of systems programming \
         experience. Built an embedded ONNX inference engine and a storage \
         layer on SQLite.\n",
    )
    .unwrap();
    std::fs::write(
        cv_dir.join("bob.txt"),
        "Bob French\nPastry chef with 12 years of experience in French \
         patisserie, sourdough baking and chocolate work.\n",
    )
    .unwrap();
    client
        .call(
            "import_cv_directory",
            &libtriqui_client::ImportCvDirectoryParams {
                job_id: job_id.into(),
                directory: cv_dir.to_string_lossy().to_string(),
            },
        )
        .expect("start import");
    let conn = open_db(&app_dir.join("triqui.db"));
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(120) {
        let extracted = scalar(
            &conn,
            &format!(
                "SELECT COUNT(*) FROM candidates WHERE job_id = '{job_id}'
                 AND extraction_status = 'Extracted'"
            ),
        );
        if extracted == 2 {
            return cv_dir;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("import of two txt CVs never finished extracting");
}

/// Temp CV root for `tag`. `~/Documents/jobs` is the app's last-resort root when
/// neither a saved preference nor `triq.yaml`/`triq.yaml` names one — and the repo
/// config leaves `gui_cv_dir_root` commented out. A serve started with
/// that config would therefore mkdir a `job-N` folder in the developer's own
/// Documents, once per `create_job`, every test run. Every serve in this file
/// pins the root here instead. Pid-scoped, so two runs never collide.
fn isolated_cv_root(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("gui-e2e-{tag}-cvroot-{}", std::process::id()))
}

/// A resume text long enough to clear the judge stage's training floor
/// (`RESUME_MIN_CHARS = 900`); shorter synthetic CVs silently fall back to
/// the cosine ad row and the judge never engages.
fn long_cv(name: &str, role: &str, skills: &str) -> String {
    let base = format!("{name}\nSenior {role} with 9 years of experience. {skills}\n");
    let filler = "\nAdditional experience: delivered production systems end to end, \
                  mentored engineers, owned on-call rotations, and wrote the design, \
                  the tests and the runbooks for every component shipped.\n";
    let mut cv = base.clone();
    while cv.chars().count() < 1200 {
        cv.push_str(filler);
    }
    cv
}

/// Open the serve app's SQLite database for inspection. WAL lets this reader
/// run alongside the live serve process.
fn open_db(path: &std::path::Path) -> rusqlite::Connection {
    let conn = rusqlite::Connection::open(path).expect("open serve db");
    conn.execute_batch("PRAGMA busy_timeout = 5000;")
        .expect("busy_timeout");
    conn
}

/// Keep new jobs' default CV dirs out of the user's real Documents folder:
/// point `AppPreferences::cv_dir_root` at a fresh temp dir — the exact rpc
/// the Settings → General → CV directory flow drives.
fn redirect_cv_root_to_temp(client: &TriqiClient, tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("gui-e2e-{tag}-cvroot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    client
        .call_typed::<serde_json::Value>(
            "save_preferences",
            &libtriqui_client::SavePreferencesParams {
                preferences: libtriqui_client::types::AppPreferences {
                    cv_dir_root: Some(root.to_string_lossy().to_string()),
                    ..Default::default()
                },
            },
        )
        .expect("save cv dir root preference");
    root
}

/// The "user restarts the app" half of a test: same `app_dir` (so the DB and
/// preferences persist) and the same pinned config for `tag`, so the restarted
/// serve cannot fall back to the real Documents folder either. Pass the tag the
/// test passed to [`spawn_serve`].
fn restart_serve(app_dir: &std::path::Path, tag: &str) -> InProcessServe {
    let cfg_path = write_config_with_isolated_cv_root(app_dir, tag);
    InProcessServe::start_with_config(app_dir.to_path_buf(), "0.0.0-e2e".to_string(), cfg_path)
        .expect("restart in-process serve")
}

fn scalar(conn: &rusqlite::Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get::<_, i64>(0))
        .expect("scalar query")
}

fn spawn_serve(tag: &str) -> (InProcessServe, TriqiClient, std::path::PathBuf) {
    let app_dir = fresh_app_dir(tag);
    let _ = std::fs::remove_dir_all(isolated_cv_root(tag));
    let cfg_path = write_config_with_isolated_cv_root(&app_dir, tag);
    let serve =
        InProcessServe::start_with_config(app_dir.clone(), "0.0.0-e2e".to_string(), cfg_path)
            .expect("start in-process serve");
    let client = serve.client();
    (serve, client, app_dir)
}

/// Like [`spawn_serve`], but also hands back the isolated CV root for
/// assertions. The judge-artifact paths are rewritten and `gui_cv_dir_root` is
/// pinned into a temp dir, so the test never touches the user's real Documents
/// folder. Returns the serve handle, the client, the isolated CV root, and the
/// pinned config path.
fn spawn_serve_isolated_cv_root(
    tag: &str,
) -> (
    InProcessServe,
    TriqiClient,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let app_dir = fresh_app_dir(tag);
    let _ = std::fs::remove_dir_all(isolated_cv_root(tag));
    let cfg_path = write_config_with_isolated_cv_root(&app_dir, tag);

    let serve =
        InProcessServe::start_with_config(app_dir, "0.0.0-e2e".to_string(), cfg_path.clone())
            .expect("start in-process serve with isolated cv root");
    let client = serve.client();
    (serve, client, isolated_cv_root(tag), cfg_path)
}

/// Write a copy of the embedded default config into `app_dir` with
/// `gui_cv_dir_root` pinned to [`isolated_cv_root`] for `tag`, and return its
/// path. Judge-artifact paths (relative to the config) are rewritten so they
/// still resolve.
///
/// Pinning goes through [`InProcessServe::start_with_config`], which scopes the
/// config to the serve thread — no process-global state, so tests can run
/// concurrently and no other serve in the process is affected. The CV root is
/// created when missing but never wiped, so a restart can reuse it.
fn write_config_with_isolated_cv_root(app_dir: &std::path::Path, tag: &str) -> std::path::PathBuf {
    let cv_root = isolated_cv_root(tag);
    std::fs::create_dir_all(&cv_root).unwrap();

    // CARGO_MANIFEST_DIR is <repo>/gui/src-tauri here (it was <repo>/tests in
    // the shared crate), so the repo root is two levels up, not one.
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    // Use the embedded default config (triq.yaml is now baked into the binary).
    // The gui_judge_artifact paths are relative to the config file's dir, so
    // rewrite them to be relative to the repo root.
    let cfg = libconfigreader::config::DEFAULT_CONFIG_YAML;
    let cfg = cfg.replace(
        "\"gui/judges/",
        &format!("\"{}/gui/judges/", repo.display()),
    );
    let cfg = format!("{cfg}\ngui_cv_dir_root: \"{}\"\n", cv_root.display());
    let cfg_path = app_dir.join("triq.yaml");
    std::fs::write(&cfg_path, cfg).unwrap();
    cfg_path
}

/// Write a config with both isolated CV root AND isolated model directory.
/// Used by tests that need a clean model cache.
fn write_config_with_isolated_model_dir(app_dir: &std::path::Path, tag: &str) -> std::path::PathBuf {
    let cv_root = isolated_cv_root(tag);
    let model_dir = std::env::temp_dir().join(format!("gui-e2e-{tag}-models-{}", std::process::id()));
    std::fs::create_dir_all(&cv_root).unwrap();
    std::fs::create_dir_all(&model_dir).unwrap();

    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let cfg = libconfigreader::config::DEFAULT_CONFIG_YAML;
    let cfg = cfg.replace(
        "\"gui/judges/",
        &format!("\"{}/gui/judges/", repo.display()),
    );
    let cfg = format!("{cfg}\ngui_cv_dir_root: \"{}\"\nmodel_dir: \"{}\"\n", cv_root.display(), model_dir.display());
    let cfg_path = app_dir.join("triq.yaml");
    std::fs::write(&cfg_path, cfg).unwrap();
    cfg_path
}

/// The consent-to-download flow's backend half: `model_variants` names the
/// arm `backend: auto` resolves to and which artifacts are on disk, without
/// loading any model. Cheap — registry resolve + filesystem checks only.
#[test]
fn model_variants_names_arm_and_presence_without_engine() {
    let (serve, client, app_dir) = spawn_serve("model-variants");

    let embedders: Vec<libtriqui_client::types::GuiEmbedder> = client
        .call_typed("list_embedders", &EmptyParams {})
        .expect("list_embedders");
    let first = embedders.first().expect("config declares gui embedders");

    let v: libtriqui_client::types::EmbedderVariants = client
        .call_typed(
            "model_variants",
            &libtriqui_client::params::ModelVariantsParams {
                model: first.slug.clone(),
            },
        )
        .expect("model_variants");
    assert_eq!(v.model, first.slug);
    assert!(
        v.runtime == "ort" || v.runtime == "mlx",
        "runtime must name a real arm, got {}",
        v.runtime
    );
    if first.downloaded {
        assert!(
            v.present,
            "downloaded embedder must have its effective arm on disk"
        );
    }

    // Unknown slug → typed refusal, not a silent guess.
    let err = client
        .call(
            "model_variants",
            &libtriqui_client::params::ModelVariantsParams {
                model: "no-such-model".into(),
            },
        )
        .expect_err("unknown slug must fail");
    assert!(err.to_string().contains("not in models registry"), "{err}");

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// The Models view's data: `list_embedders` (on-disk state enriched) and the
/// vetted-embedder cards; the default-model toggle persists to preferences
/// and survives a restart.
#[test]
fn embedder_lists_contract_and_default_model_persists() {
    let (serve, client, app_dir) = spawn_serve("models");

    let embedders: Vec<libtriqui_client::types::GuiEmbedder> = client
        .call_typed("list_embedders", &EmptyParams {})
        .expect("list_embedders");
    assert!(
        !embedders.is_empty(),
        "config must define gui_embedders"
    );
    assert_eq!(
        embedders.iter().filter(|e| e.is_default).count(),
        1,
        "exactly one default embedder on a fresh install"
    );
    for e in &embedders {
        assert!(!e.slug.is_empty() && !e.name.is_empty(), "identity fields");
        assert!(e.dim > 0 && e.max_context_window > 0 && e.memory_mb > 0);
        assert!(
            e.source.starts_with("hf:"),
            "hugging face source: {}",
            e.source
        );
        assert_eq!(
            e.size_bytes > 0,
            e.downloaded,
            "{}: size/downloaded consistency",
            e.slug
        );
    }

    // Vetted cards (manually vetted, with real system RAM).
    let vetted: libtriqui_client::types::VettedEmbeddersResponse = client
        .call_typed("list_vetted_embedders", &EmptyParams {})
        .expect("list_vetted_embedders");
    assert!(!vetted.embedders.is_empty());
    assert!(vetted.system_ram_mb > 0, "real system RAM reported");
    assert!(!vetted.recommended_slug.is_empty());
    for v in &vetted.embedders {
        assert!(!v.slug.is_empty());
        assert!(v.memory_mb > 0);
        assert!(v.dim > 0);
        assert!(v.max_context_window > 0);
    }

    // The default toggle: flips is_default and persists to preferences.
    let default_slug = embedders
        .iter()
        .find(|e| e.is_default)
        .expect("a default embedder")
        .slug
        .clone();
    let other_slug = embedders
        .iter()
        .find(|e| !e.is_default)
        .expect("a second gui embedder")
        .slug
        .clone();
    client
        .call(
            "set_default_model",
            &libtriqui_client::SetDefaultModelParams {
                kind: libtriqui_client::types::ModelKind::Embedding,
                repo_id: other_slug.clone(),
            },
        )
        .expect("set default model");
    let fresh: Vec<libtriqui_client::types::GuiEmbedder> = client
        .call_typed("list_embedders", &EmptyParams {})
        .expect("re-list embedders");
    let is_default = |slug: &str| {
        fresh
            .iter()
            .find(|e| e.slug == slug)
            .unwrap_or_else(|| panic!("{slug} listed"))
            .is_default
    };
    assert!(is_default(&other_slug), "the new default is reported");
    assert!(!is_default(&default_slug), "the old default is cleared");
    let prefs: libtriqui_client::types::AppPreferences = client
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load preferences");
    assert_eq!(
        prefs.embedding_model.as_deref(),
        Some(other_slug.as_str()),
        "the default model is the saved embedding_model preference"
    );

    // Persists across a restart.
    serve.shutdown();
    let serve2 = restart_serve(&app_dir, "models");
    let fresh2: Vec<libtriqui_client::types::GuiEmbedder> = serve2
        .client()
        .call_typed("list_embedders", &EmptyParams {})
        .expect("list embedders after restart");
    assert!(
        fresh2
            .iter()
            .find(|e| e.slug == other_slug)
            .expect("new default after restart")
            .is_default,
        "the default-model choice survives the restart"
    );

    serve2.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// A stale `running` cascade row on restart must be recovered as cancelled.
/// The `running` flag in `rank_state` is only truthful while the process that
/// spawned the cascade lives. A quit or crash mid-cascade leaves the row
/// behind, and the next launch inherits a job that reports running forever
/// and refuses every new cascade — the wedged-job bug. Serve start must
/// recover the stale row.
///
/// Cheap: the row is written directly; no cascade ever runs.
#[test]
fn a_stale_running_cascade_state_recovers_on_serve_restart() {
    let (serve, client, app_dir) = spawn_serve("wedgerecover");
    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");
    let job_id = job.id.clone();

    // Simulate the crash: a run in flight, as `spawn_cascade` writes it.
    {
        let conn = open_db(&app_dir.join("triqui.db"));
        conn.execute(
            "INSERT INTO rank_state (job_id, sort, sort_reports_json, complete, cancelled)
             VALUES (?1, 'sort', '[]', 0, 0)",
            rusqlite::params![job_id],
        )
        .expect("seed wedged rank_state");
    }
    let wedged: libtriqui_client::types::RankState = client
        .call_typed(
            "get_rank_state",
            &StartRankParams {
                job_id: job_id.clone(),
            },
        )
        .expect("rank state");
    assert!(
        wedged.running,
        "pre-condition: the seeded row reads running"
    );

    // The user restarts the app. Serve start owns recovery: no cascade thread
    // exists yet, so the row cannot be telling the truth.
    serve.shutdown();
    let serve2 = restart_serve(&app_dir, "wedgerecover");
    let client2 = serve2.client();
    let recovered: libtriqui_client::types::RankState = client2
        .call_typed(
            "get_rank_state",
            &StartRankParams {
                job_id: job_id.clone(),
            },
        )
        .expect("rank state after restart");

    assert!(!recovered.running, "the stale run must not read running");
    assert!(
        !recovered.converged,
        "recovery is a cancel, not a convergence"
    );
    assert_eq!(
        recovered.reason.as_deref(),
        Some("cancelled"),
        "recovered state reports the run as cancelled"
    );

    serve2.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// The new user-driven model selection flow: 4 variants (270M/600M × ONNX/MLX)
/// are listed, none are pre-downloaded, no startup modal, engine stays Stopped
/// until the user downloads and explicitly selects a model via set_embedding_model.
#[test]
fn user_driven_model_selection_flow() {
    let app_dir = fresh_app_dir("model-selection");
    let cfg_path = write_config_with_isolated_model_dir(&app_dir, "model-selection");
    let serve = InProcessServe::start_with_config(app_dir.clone(), "0.0.0-e2e".to_string(), cfg_path)
        .expect("start in-process serve");
    let client = serve.client();

    // 1. list_embedders returns 4 variants, all not downloaded, exactly one default
    let embedders: Vec<libtriqui_client::types::GuiEmbedder> = client
        .call_typed("list_embedders", &EmptyParams {})
        .expect("list_embedders");

    assert_eq!(embedders.len(), 4, "should have 4 model variants");

    let slugs: Vec<&str> = embedders.iter().map(|e| e.slug.as_str()).collect();
    assert!(slugs.contains(&"harrier-oss-v1-270m-ONNX-q4f16"), "should have 270M ONNX");
    assert!(slugs.contains(&"harrier-oss-v1-270m-MLX-4bit"), "should have 270M MLX");
    assert!(slugs.contains(&"harrier-oss-v1-0.6b-ONNX-q4"), "should have 600M ONNX");
    assert!(slugs.contains(&"harrier-oss-v1-0.6b-MLX-4bit"), "should have 600M MLX");

    let default_count = embedders.iter().filter(|e| e.is_default).count();
    assert_eq!(default_count, 1, "exactly one default");
    let default = embedders.iter().find(|e| e.is_default).unwrap();
    assert_eq!(default.slug, "harrier-oss-v1-270m-ONNX-q4f16", "270M ONNX is default");

    // All should be not downloaded on fresh install
    for e in &embedders {
        assert!(!e.downloaded, "{} should not be downloaded on fresh install", e.slug);
        assert_eq!(e.size_bytes, 0, "{} should have 0 size_bytes", e.slug);
    }

    // 2. Engine status should be Stopped (no auto-start)
    let status: libtriqui_client::types::RuntimeStatusInfo = client
        .call_typed("runtime_status", &EmptyParams {})
        .expect("runtime_status");
    assert_eq!(status.status, "Stopped", "engine should be Stopped on fresh launch");

    // 3. Download one model (270M ONNX)
    client
        .call_typed::<serde_json::Value>(
            "download_embedder",
            &libtriqui_client::params::DownloadEmbedderParams {
                slug: "harrier-oss-v1-270m-ONNX-q4f16".into(),
            },
        )
        .expect("download 270M ONNX");

    // 4. Verify it's now downloaded
    let embedders: Vec<libtriqui_client::types::GuiEmbedder> = client
        .call_typed("list_embedders", &EmptyParams {})
        .expect("list_embedders after download");
    let downloaded = embedders.iter().find(|e| e.slug == "harrier-oss-v1-270m-ONNX-q4f16").unwrap();
    assert!(downloaded.downloaded, "270M ONNX should be downloaded");
    assert!(downloaded.size_bytes > 0, "should have non-zero size_bytes");

    // 5. Switch to the downloaded model (set_embedding_model hot-swaps, engine stays Stopped until first embed)
    client
        .call_typed::<serde_json::Value>(
            "set_embedding_model",
            &libtriqui_client::params::SetEmbeddingModelParams {
                slug: "harrier-oss-v1-270m-ONNX-q4f16".into(),
            },
        )
        .expect("set embedding model");

    // 6. Engine should still be Stopped (lazy start on first embed)
    let status: libtriqui_client::types::RuntimeStatusInfo = client
        .call_typed("runtime_status", &EmptyParams {})
        .expect("runtime_status after switch");
    assert_eq!(status.status, "Stopped", "engine should be Stopped after set_embedding_model (lazy start)");
    assert_eq!(status.model.as_deref(), Some("harrier-oss-v1-270m-ONNX-q4f16"));

    // 7. Preference should be persisted
    let prefs: libtriqui_client::types::AppPreferences = client
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load preferences");
    assert_eq!(prefs.embedding_model.as_deref(), Some("harrier-oss-v1-270m-ONNX-q4f16"));

    // 8. Switch to another model (600M ONNX) - download + switch
    client
        .call_typed::<serde_json::Value>(
            "download_embedder",
            &libtriqui_client::params::DownloadEmbedderParams {
                slug: "harrier-oss-v1-0.6b-ONNX-q4".into(),
            },
        )
        .expect("download 600M ONNX");

    client
        .call_typed::<serde_json::Value>(
            "set_embedding_model",
            &libtriqui_client::params::SetEmbeddingModelParams {
                slug: "harrier-oss-v1-0.6b-ONNX-q4".into(),
            },
        )
        .expect("set embedding model to 600M ONNX");

    let status: libtriqui_client::types::RuntimeStatusInfo = client
        .call_typed("runtime_status", &EmptyParams {})
        .expect("runtime_status after second switch");
    assert_eq!(status.model.as_deref(), Some("harrier-oss-v1-0.6b-ONNX-q4"));

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// Old slug migration: existing users with old slug should have it mapped to new slug.
#[test]
fn old_slug_migration() {
    let app_dir = fresh_app_dir("slug-migration");
    let cfg_path = write_config_with_isolated_cv_root(&app_dir, "slug-migration");
    let serve = InProcessServe::start_with_config(app_dir.clone(), "0.0.0-e2e".to_string(), cfg_path)
        .expect("start in-process serve");
    let client = serve.client();

    // Manually write old preferences with old slug
    let prefs_path = app_dir.join("preferences.json");
    let old_prefs = serde_json::json!({
        "theme": "system",
        "color_theme": "default",
        "quick_pane_shortcut": null,
        "language": null,
        "embedding_model": "harrier-oss-v1-270m-ONNX_model_q4f16",
        "inference_model": null,
        "cv_dir_root": null,
        "engine_backend": null,
        "engine_exec_provider": null,
        "engine_intra_threads": null,
        "engine_inter_threads": null,
        "engine_embed_batch": null,
        "engine_max_cores": null,
        "engine_no_cache": null
    });
    std::fs::write(&prefs_path, serde_json::to_string_pretty(&old_prefs).unwrap()).unwrap();

    // Restart serve to load migrated preferences
    serve.shutdown();
    let serve2 = restart_serve(&app_dir, "slug-migration");
    let client = serve2.client();

    let prefs: libtriqui_client::types::AppPreferences = client
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load preferences after migration");

    assert_eq!(prefs.embedding_model.as_deref(), Some("harrier-oss-v1-270m-ONNX-q4f16"));

    serve2.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}