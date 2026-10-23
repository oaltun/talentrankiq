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