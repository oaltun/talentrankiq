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

#[test]
fn job_lifecycle_crud_archive_delete_and_persistence() {
    let (serve, client, app_dir) = spawn_serve("jobcrud");
    let cv_root = redirect_cv_root_to_temp(&client, "jobcrud");

    // Create two jobs; shape the first (rename, size, ad, weights).
    let job1: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job 1");
    // Rubric weights default from the embedded config's `rubric_defaults`.
    // Read it so the test tracks the config.
    let rubric = &libconfigreader::Config::default_from_embedded(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    )
    .rubric_defaults;
    assert_eq!(job1.ad_weight, rubric.ad_points);
    assert_eq!(job1.fb_job_weight, rubric.fb_job_points);
    assert_eq!(job1.fb_profile_weight, rubric.fb_profile_points);
    assert!(
        job1.inputs_updated_at.is_some(),
        "a fresh job's inputs count as changed: the re-rank is due"
    );
    let job2: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job 2");

    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job1.id.clone(),
                title: "Rust Engineer".into(),
                job_ad_text: Some("Build fast things.".into()),
                triqui_size: 3,
            },
        )
        .expect("shape job 1");
    client
        .call(
            "update_job_weights",
            &libtriqui_client::UpdateJobWeightsParams {
                job_id: job1.id.clone(),
                ad_weight: 40.0,
                fb_job_weight: 20.0,
                fb_profile_weight: 15.0,
            },
        )
        .expect("update weights");

    // Summaries carry the precomputed facts the Jobs list renders.
    let summaries = |client: &TriqiClient| -> Vec<libtriqui_client::types::JobSummary> {
        client
            .call_typed("list_job_summaries", &EmptyParams {})
            .expect("list_job_summaries")
    };
    let s1 = summaries(&client)
        .into_iter()
        .find(|s| s.id == job1.id)
        .expect("job 1 summary");
    assert_eq!(s1.title, "Rust Engineer");
    assert_eq!(s1.triqui_size, 3);
    assert!(s1.ad_set, "stored ad text reported as set");
    assert_eq!((s1.cv_imported, s1.cv_failed), (0, 0));
    assert_eq!((s1.want_count, s1.avoid_count), (0, 0));
    assert!(!s1.archived);
    assert!(s1.last_ranked_at.is_none());

    // Archive + restore (the Jobs list's archive toggle).
    client
        .call(
            "archive_job",
            &libtriqui_client::ArchiveJobParams {
                job_id: job2.id.clone(),
                archived: true,
            },
        )
        .expect("archive job 2");
    let s2 = summaries(&client)
        .into_iter()
        .find(|s| s.id == job2.id)
        .expect("job 2 summary");
    assert!(s2.archived, "archived flag reported");
    client
        .call(
            "archive_job",
            &libtriqui_client::ArchiveJobParams {
                job_id: job2.id.clone(),
                archived: false,
            },
        )
        .expect("restore job 2");
    let s2 = summaries(&client)
        .into_iter()
        .find(|s| s.id == job2.id)
        .expect("job 2 summary");
    assert!(!s2.archived, "restored job is active again");

    // Everything survives an app restart.
    serve.shutdown();
    let serve2 = restart_serve(&app_dir, "jobcrud");
    let client = serve2.client();
    let detail: libtriqui_client::types::JobDetail = client
        .call_typed(
            "get_job_detail",
            &GetJobDetailParams {
                job_id: job1.id.clone(),
            },
        )
        .expect("job 1 after restart");
    assert_eq!(detail.job.title, "Rust Engineer");
    assert_eq!(detail.job.triqui_size, 3);
    assert_eq!(
        detail.job.job_ad_text.as_deref(),
        Some("Build fast things.")
    );
    assert_eq!(
        (
            detail.job.ad_weight,
            detail.job.fb_job_weight,
            detail.job.fb_profile_weight
        ),
        (40.0, 20.0, 15.0),
        "weights persist across the restart"
    );
    let s2 = summaries(&client)
        .into_iter()
        .find(|s| s.id == job2.id)
        .expect("job 2 after restart");
    assert!(!s2.archived);

    // Delete removes the job for good.
    client
        .call(
            "delete_job",
            &libtriqui_client::DeleteJobParams {
                job_id: job2.id.clone(),
            },
        )
        .expect("delete job 2");
    let de = expect_domain(
        &client,
        "get_job_detail",
        &GetJobDetailParams {
            job_id: job2.id.clone(),
        },
        "get deleted job",
    );
    assert!(matches!(de, DomainError::NotFound { ref id, .. } if *id == job2.id));
    assert!(
        !summaries(&client).iter().any(|s| s.id == job2.id),
        "deleted job vanishes from the list"
    );

    serve2.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
}

/// The job ad lifecycle behind the Setup pane. `update_job` always writes the
/// full form, so `job_ad_text: None` CLEARS the ad (the Clear button), and an
/// edit that leaves ad + size untouched must not bump `inputs_updated_at`
/// (the stale badge) — while a weight change must.
#[test]
fn job_ad_text_and_input_change_semantics() {
    let (serve, client, app_dir) = spawn_serve("adtext");
    let cv_root = redirect_cv_root_to_temp(&client, "adtext");
    let job_id = create_job_with_ad(&client, "Engineer", "First ad version.");

    let job = || -> libtriqui_client::types::Job {
        client
            .call_typed::<libtriqui_client::types::JobDetail>(
                "get_job_detail",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("job detail")
            .job
    };
    assert_eq!(job().job_ad_text.as_deref(), Some("First ad version."));
    let ad_set = |client: &TriqiClient| {
        client
            .call_typed::<Vec<libtriqui_client::types::JobSummary>>(
                "list_job_summaries",
                &EmptyParams {},
            )
            .expect("summaries")
            .into_iter()
            .find(|s| s.id == job_id)
            .expect("summary")
            .ad_set
    };
    assert!(ad_set(&client));

    // Clear (the Clear button sends None) → ad gone, ad_set false.
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Engineer".into(),
                job_ad_text: None,
                triqui_size: 10,
            },
        )
        .expect("clear ad");
    assert_eq!(job().job_ad_text, None, "None clears the stored ad");
    assert!(!ad_set(&client));

    // Re-set, then a title-only edit (same ad + size): the stale-relevant
    // inputs did not change → inputs_updated_at untouched.
    let inputs_after_set = job().inputs_updated_at.expect("ad set bumps inputs");
    std::thread::sleep(Duration::from_millis(1100)); // cross a datetime('now') tick
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Senior Engineer".into(),
                job_ad_text: Some("Second ad version.".into()),
                triqui_size: 10,
            },
        )
        .expect("re-set ad");
    let inputs_before = job().inputs_updated_at.expect("ad re-set bumps inputs");
    assert!(
        inputs_before > inputs_after_set,
        "changing the ad is a ranking-input change"
    );
    std::thread::sleep(Duration::from_millis(1100));
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Principal Engineer".into(),
                job_ad_text: Some("Second ad version.".into()),
                triqui_size: 10,
            },
        )
        .expect("title-only edit");
    assert_eq!(
        job().inputs_updated_at,
        Some(inputs_before.clone()),
        "a mere title edit must not bump inputs_updated_at"
    );

    // A weight change is a ranking-input change → bumped.
    std::thread::sleep(Duration::from_millis(1100));
    client
        .call(
            "update_job_weights",
            &libtriqui_client::UpdateJobWeightsParams {
                job_id: job_id.clone(),
                ad_weight: 40.0,
                fb_job_weight: 10.0,
                fb_profile_weight: 10.0,
            },
        )
        .expect("weight change");
    let inputs_after_weights = job().inputs_updated_at.expect("weights bump inputs");
    assert!(
        inputs_after_weights > inputs_before,
        "weight change must bump inputs_updated_at"
    );

    // Validation surfaces as typed errors before anything is written.
    let de = expect_domain(
        &client,
        "update_job",
        &libtriqui_client::UpdateJobParams {
            job_id: job_id.clone(),
            title: "   ".into(),
            job_ad_text: None,
            triqui_size: 10,
        },
        "empty title",
    );
    assert!(matches!(de, DomainError::ValidationError { .. }));
    let de = expect_domain(
        &client,
        "update_job",
        &libtriqui_client::UpdateJobParams {
            job_id: job_id.clone(),
            title: "Engineer".into(),
            job_ad_text: None,
            triqui_size: 0,
        },
        "triqui size 0",
    );
    assert!(matches!(de, DomainError::ValidationError { .. }));

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
}

/// The gui's Re-rank button is a projection of backend state: `job_ui_state`
/// must report the domain truth (allowed / dirty / reason code) with no engine
/// involved. Cheap — runs even with `IK_SKIP_EXPENSIVE=1`.
#[test]
fn job_ui_state_reason_codes_without_engine() {
    let (serve, client, app_dir) = spawn_serve("ui-state");

    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");

    // Fresh job: nothing imported → rank is blocked with a stable reason
    // code; never-ranked inputs count as dirty (a rank is due).
    let st: libtriqui_client::types::JobUiState = client
        .call_typed(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: job.id.clone(),
            },
        )
        .expect("job_ui_state");
    assert!(!st.sort_allowed, "no CVs imported yet");
    assert!(st.rerank_dirty, "never-ranked inputs need a rank");
    assert_eq!(st.reason.as_deref(), Some("no-cvs"));

    // Unknown job → typed NotFound.
    let err = client
        .call(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: "job_missing".into(),
            },
        )
        .expect_err("unknown job must fail");
    let ClientError::Domain(pe) = err else {
        panic!("expected domain error, got {err:?}");
    };
    let de: DomainError = serde_json::from_value(pe.domain.expect("domain payload")).unwrap();
    assert!(matches!(de, DomainError::NotFound { ref id, .. } if id == "job_missing"));

    // No rank has run → no sort report exists yet (typed NotFound).
    let err = client
        .call(
            "get_sort_report",
            &GetJobDetailParams {
                job_id: job.id.clone(),
            },
        )
        .expect_err("no report before first rank");
    let ClientError::Domain(pe) = err else {
        panic!("expected domain error, got {err:?}");
    };
    let de: DomainError = serde_json::from_value(pe.domain.expect("domain payload")).unwrap();
    assert!(matches!(de, DomainError::NotFound { ref id, .. } if *id == job.id));

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// `no-jobad` is a backend-derived blocker, not gui styling: with CVs
/// extracted but no job ad text, `job_ui_state` reports the reason and
/// `start_rank` refuses server-side (the old behaviour ranked anyway and
/// produced all-zero totals). Cheap — extraction needs no engine.
#[test]
fn no_jobad_blocks_rank_with_reason_code() {
    let (serve, client, app_dir) = spawn_serve("no-jobad");

    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");
    let job_id = job.id.clone();

    // CVs only — no ad via update_job.
    let cv_dir = app_dir.join("cvs-nojobad");
    std::fs::create_dir_all(&cv_dir).unwrap();
    std::fs::write(
        cv_dir.join("alice.txt"),
        long_cv("Alice Chen", "Java engineer", "JVM, Spring, throughput."),
    )
    .unwrap();

    client
        .call(
            "import_cv_directory",
            &libtriqui_client::ImportCvDirectoryParams {
                job_id: job_id.clone(),
                directory: cv_dir.to_string_lossy().to_string(),
            },
        )
        .expect("start import");

    // Poll until extraction lands: the reason must move no-cvs → no-jobad
    // (never to engine-not-ready/stale/clean, which would mean ranking is
    // considered allowed without an ad).
    let mut state: Option<libtriqui_client::types::JobUiState> = None;
    for _ in 0..120 {
        std::thread::sleep(Duration::from_millis(500));
        let st: libtriqui_client::types::JobUiState = client
            .call_typed(
                "job_ui_state",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("job_ui_state");
        if st.reason.as_deref() != Some("no-cvs") {
            state = Some(st);
            break;
        }
    }
    let st = state.expect("extraction never moved past no-cvs");
    assert!(!st.sort_allowed, "no job ad → rank must stay blocked");
    assert_eq!(st.reason.as_deref(), Some("no-jobad"));

    // The import auto-started an embed-prefill cascade (design §9). It is
    // background embedding, not a ranking: the button-facing `ranking` flag
    // must stay off for as long as the job has no ad — through the in-flight
    // window (`running` true) and after the run settles. The rank buttons
    // spin on `ranking`; spinning here was the "Ranking… with nothing to
    // rank" bug.
    for _ in 0..20 {
        let s: libtriqui_client::types::RankState = client
            .call_typed(
                "get_rank_state",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("rank state");
        assert!(
            !s.ranking,
            "no job ad → the cascade must never report ranking=true"
        );
        std::thread::sleep(Duration::from_millis(250));
    }

    // The convergence nudge (`job_ui_state.nudge = add-job-ad`) must fire
    // only on a *settled* ad-less embed-prefill run — never mid-flight and
    // never for a cancelled one. Whether this environment converges at all
    // depends on the engine (cheap runs error out; full runs may embed for
    // real), so pin the race-safe direction here: a nudge implies the run
    // converged. The Some/None lifecycle itself is pinned deterministically
    // by the full-workflow ad-lands test below.
    let ui: libtriqui_client::types::JobUiState = client
        .call_typed(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("job_ui_state");
    if ui.nudge.as_deref() == Some("add-job-ad") {
        let rs: libtriqui_client::types::RankState = client
            .call_typed(
                "get_rank_state",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("rank state");
        assert!(
            rs.converged && !rs.running,
            "nudge fired without a converged prefill run (rank_state: {rs:?})"
        );
    }

    // Server-side refusal, not just a button state: the cascade gate rejects
    // the start command (and the import auto-start logged the same refusal).
    let err = client
        .call(
            "start_rank",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect_err("rank without a job ad must be refused");
    assert!(
        err.to_string().contains("no job ad"),
        "refusal must name the missing ad, got: {err}"
    );

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// Settings preferences round-trip through serve and persist to
/// `<app_dir>/preferences.json` across an app restart.
#[test]
fn preferences_round_trip_and_restart_persistence() {
    let (serve, client, app_dir) = spawn_serve("prefs");

    let saved = libtriqui_client::types::AppPreferences {
        theme: "dark".into(),
        color_theme: "ocean".into(),
        quick_pane_shortcut: Some("Ctrl+Alt+K".into()),
        language: Some("de".into()),
        embedding_model: Some("nomic-embed-text-v1.5".into()),
        inference_model: Some("test-model".into()),
        cv_dir_root: Some(app_dir.join("cv-root").to_string_lossy().to_string()),
        ..Default::default()
    };
    client
        .call_typed::<serde_json::Value>(
            "save_preferences",
            &libtriqui_client::SavePreferencesParams {
                preferences: saved.clone(),
            },
        )
        .expect("save preferences");
    let json = |p: &libtriqui_client::types::AppPreferences| {
        serde_json::to_value(p).expect("serialize preferences")
    };
    let loaded: libtriqui_client::types::AppPreferences = client
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load preferences");
    assert_eq!(json(&loaded), json(&saved), "saved preferences read back");

    serve.shutdown();
    let serve2 = restart_serve(&app_dir, "prefs");
    let loaded2: libtriqui_client::types::AppPreferences = serve2
        .client()
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load preferences after restart");
    assert_eq!(
        json(&loaded2),
        json(&saved),
        "preferences persist across the restart"
    );

    // Reset to defaults (the Settings reset buttons).
    serve2
        .client()
        .call_typed::<serde_json::Value>(
            "save_preferences",
            &libtriqui_client::SavePreferencesParams {
                preferences: libtriqui_client::types::AppPreferences::default(),
            },
        )
        .expect("reset preferences");
    let reset: libtriqui_client::types::AppPreferences = serve2
        .client()
        .call_typed("load_preferences", &EmptyParams {})
        .expect("load reset preferences");
    assert_eq!(reset.theme, "system");
    assert_eq!(reset.color_theme, "default");
    assert_eq!(reset.cv_dir_root, None);

    serve2.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
}

/// Settings → General → CV directory: the gui writes
/// `AppPreferences::cv_dir_root` into `preferences.json` and new jobs' default
/// CV dirs must follow it, overriding the config's `gui_cv_dir_root`. Reset
/// (null) falls back to the config root. Cheap: rpcs only, no engine.
#[test]
fn cv_dir_root_preference_overrides_config_for_new_jobs() {
    let (serve, client, cv_root, _cfg_path) = spawn_serve_isolated_cv_root("cvrootpref");

    // No preference saved yet → the effective root is the config one.
    let effective: String = client
        .call_typed("get_cv_dir_root", &EmptyParams {})
        .expect("get_cv_dir_root");
    assert_eq!(
        effective,
        cv_root.to_string_lossy(),
        "unset preference falls back to the configured root"
    );

    let custom_root =
        std::env::temp_dir().join(format!("gui-e2e-cvrootpref-custom-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&custom_root);

    // Save a custom root (what the Settings dialog's folder picker writes).
    client
        .call_typed::<serde_json::Value>(
            "save_preferences",
            &libtriqui_client::SavePreferencesParams {
                preferences: libtriqui_client::types::AppPreferences {
                    cv_dir_root: Some(custom_root.to_string_lossy().to_string()),
                    ..Default::default()
                },
            },
        )
        .expect("save preferences with custom cv root");
    let effective: String = client
        .call_typed("get_cv_dir_root", &EmptyParams {})
        .expect("get_cv_dir_root after save");
    assert_eq!(effective, custom_root.to_string_lossy());

    // A job created after the change defaults under the custom root.
    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job with custom root");
    let dir = job.import_directory_path.expect("default cv dir");
    assert!(
        dir.starts_with(custom_root.to_string_lossy().as_ref()),
        "new job's default CV dir {dir:?} must live under the preferred root"
    );

    // Reset to default (the Settings "reset" button) → back to the config root.
    client
        .call_typed::<serde_json::Value>(
            "save_preferences",
            &libtriqui_client::SavePreferencesParams {
                preferences: libtriqui_client::types::AppPreferences::default(),
            },
        )
        .expect("reset preferences");
    let job2: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job after reset");
    let dir2 = job2.import_directory_path.expect("default cv dir");
    assert!(
        dir2.starts_with(cv_root.to_string_lossy().as_ref()),
        "after reset the default CV dir {dir2:?} falls back to the configured root"
    );

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&cv_root);
    let _ = std::fs::remove_dir_all(&custom_root);
}

/// The results sheet is a projection of per-stage score rows: one column per
/// cascade step — scout (0.5) → job ad (heavy embedder) → judge — and each candidate carries exactly the grades the stages have given
/// it. A stage that has not reached a candidate has **no row** for it, which is
/// what the gui renders as `–`; the columns themselves are always present, so
/// the sheet does not reshuffle as waves land.
///
/// Cheap and engine-free: the stage rows are seeded directly (an import would
/// start a real cascade, which is what the judge workflow test covers).
#[test]
fn results_table_shows_a_grade_per_cascade_stage() {
    let (serve, client, app_dir) = spawn_serve("stages");
    let cv_root = redirect_cv_root_to_temp(&client, "stages");
    let job_id = create_job_with_ad(&client, "Stage Engineer", "Rust backend engineer.");

    let detail: libtriqui_client::types::JobDetail = client
        .call_typed(
            "get_job_detail",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("job detail");
    let ids: Vec<&str> = detail
        .rubric
        .iter()
        .map(|q| q.question_id.as_str())
        .collect();
    for stage in ["scout", "ad", "ad_judge"] {
        assert!(
            ids.contains(&stage),
            "the results table has a column for the '{stage}' stage: {ids:?}"
        );
    }
    // Stage columns come first, in the order the pipeline produces them.
    assert_eq!(&ids[..3], ["scout", "ad", "ad_judge"]);
    let kinds: Vec<&str> = detail
        .rubric
        .iter()
        .take(3)
        .map(|q| q.kind.as_str())
        .collect();
    assert_eq!(kinds, ["Scout", "Ad", "AdJudge"]);
    // Every stage is scored in the job ad's currency, so the columns are
    // comparable with the ad column they sit next to.
    let ad_weight = detail.job.ad_weight as f32;
    assert!(
        detail
            .rubric
            .iter()
            .take(3)
            .all(|q| (q.max_points - ad_weight).abs() < 1e-4),
        "stage columns are on the ad scale"
    );
    // The Total caption counts the rubric only — stage grades are not slices of it.
    let expected_capacity: f32 = detail
        .rubric
        .iter()
        .filter(|q| !q.kind.is_stage())
        .map(|q| q.max_points.abs())
        .sum();
    assert!(
        (detail.rubric_capacity - expected_capacity).abs() < 1e-4,
        "rubric capacity {} excludes the stage columns",
        detail.rubric_capacity
    );

    // Two candidates: one the whole cascade has graded, one scout-only.
    {
        let conn = open_db(&app_dir.join("triqui.db"));
        for (id, name) in [("cand_full", "full.pdf"), ("cand_early", "early.pdf")] {
            conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, extracted_text)
                 VALUES (?1, ?2, ?3, ?4, 'Pdf', 'Extracted', 'some cv text')",
                rusqlite::params![id, job_id, format!("/tmp/{id}"), name],
            )
            .expect("candidate");
        }
        // Only the fully-graded candidate has a total (the judge's own row).
        conn.execute(
            "UPDATE candidates SET points = 40.0, points_max = 40.0 WHERE id = 'cand_full'",
            [],
        )
        .expect("points");
        let rows = [
            ("cand_full", "scout", "Scout", 26.0),
            ("cand_full", "ad", "Ad", 21.0),
            ("cand_full", "ad_judge", "AdJudge", 40.0),
            ("cand_early", "scout", "Scout", 10.0),
        ];
        for (cid, qid, kind, pts) in rows {
            conn.execute(
                "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
                 VALUES (?1, ?2, ?3, ?4, 'Positive', 'label', ?5, 40.0)",
                rusqlite::params![format!("cs_{cid}_{qid}"), cid, qid, kind, pts],
            )
            .expect("score row");
        }
    }

    let page = |sort_by: Option<&str>| -> libtriqui_client::types::CandidatePage {
        client
            .call_typed(
                "list_candidates",
                &libtriqui_client::ListCandidatesParams {
                    job_id: job_id.clone(),
                    offset: 0,
                    limit: 50,
                    query: None,
                    sort_by: sort_by.map(str::to_string),
                    sort_dir: Some(false),
                },
            )
            .expect("list_candidates")
    };
    let scores_of = |page: &libtriqui_client::types::CandidatePage, id: &str| {
        page.items
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("{id} in page"))
            .scores
            .iter()
            .map(|s| (s.question_id.clone(), s.kind.as_str().to_string()))
            .collect::<Vec<_>>()
    };

    // Every stage a candidate was graded on is present, each under its own id,
    // in the sheet's order (ad → judge → the stage grades).
    let all = page(None);
    assert_eq!(
        scores_of(&all, "cand_full"),
        vec![
            ("ad".to_string(), "Ad".to_string()),
            ("ad_judge".to_string(), "AdJudge".to_string()),
            ("scout".to_string(), "Scout".to_string()),
        ]
    );
    // A candidate the later stages have not reached simply has no row for them.
    assert_eq!(
        scores_of(&all, "cand_early"),
        vec![("scout".to_string(), "Scout".to_string())]
    );

    // Two independent quality axes (scoring v3), not one conflated chip: the
    // candidate the heavy embedder reached is in the heavy space whatever backs
    // its ad slot, and what backs the ad slot is the judge's win-rate.
    assert_eq!(
        all.items[0].grade_space,
        Some(libtriqui_client::types::GradeSpace::Heavy)
    );
    assert_eq!(
        all.items[0].ad_source,
        Some(libtriqui_client::types::AdSource::Judge)
    );
    assert_eq!(
        all.items[1].grade_space,
        Some(libtriqui_client::types::GradeSpace::Scout)
    );
    assert_eq!(
        all.items[1].ad_source,
        Some(libtriqui_client::types::AdSource::Scout)
    );
    // The default order leads with the better space, then the total inside it —
    // so the heavy-space candidate outranks the scout-only one regardless of the
    // raw numbers, which is the point of the tier.
    assert_eq!(all.items[0].id, "cand_full", "heavy space leads");
    assert_eq!(all.items[1].id, "cand_early", "scout space follows");
    // `__source` is the ad-slot axis on its own: judge first, then heavy, then scout.
    let by_source = page(Some("__source"));
    assert_eq!(by_source.items[0].id, "cand_full", "judge tier leads");
    assert_eq!(by_source.items[1].id, "cand_early", "scout tier follows");

    // Each stage column is a sort control, and ungraded candidates sort last.
    assert_eq!(
        all.items[0].id, "cand_full",
        "default order: the heavy-space candidate leads"
    );
    let by_scout = page(Some("scout"));
    assert_eq!(by_scout.items[0].id, "cand_full", "scout grade leads");
    assert_eq!(
        by_scout.items[1].id, "cand_early",
        "unscouted candidate sorts last"
    );
    let by_judge = page(Some("ad_judge"));
    assert_eq!(by_judge.items[0].id, "cand_full", "judge grade leads");
    assert_eq!(
        by_judge.items[1].id, "cand_early",
        "unjudged candidate sorts last"
    );

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
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

/// The ad landing **mid-run** (the user pastes it while the pre-ad cascade is
/// still embedding) must not cancel the run: its embeds are ad-independent and
/// its next wave picks the ad up — the sort scores from the stored chunks,
/// the judge judges. The run then
/// converges straight through, without a cancel/restart cycle.
///
/// Full-workflow: real engine, real model cache — expensive-test pattern.
#[test]
fn an_ad_landing_mid_run_is_picked_up_without_cancelling_the_cascade() {
    if std::env::var("IK_SKIP_EXPENSIVE").is_ok() {
        eprintln!("skipping: expensive (real engine, real model cache)");
        return;
    }
    let (serve, client, app_dir) = spawn_serve("midrun");
    let cv_root = redirect_cv_root_to_temp(&client, "midrun");
    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");
    let job_id = job.id.clone();
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Mid-Run Ad Engineer".into(),
                job_ad_text: None,
                triqui_size: 10,
            },
        )
        .expect("set title");
    import_two_txt_cvs(&client, &cv_root, &app_dir, &job_id);

    // Wait until the run is actually in flight (the rank_state row exists and
    // reads running), then land the ad under it.
    let start = Instant::now();
    loop {
        let running = client
            .call_typed::<libtriqui_client::types::RankState>(
                "get_rank_state",
                &StartRankParams {
                    job_id: job_id.clone(),
                },
            )
            .map(|s| s.running)
            .unwrap_or(false);
        if running {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "the pre-ad cascade never started"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Mid-Run Ad Engineer".into(),
                job_ad_text: Some(
                    "Senior Rust engineer with systems programming experience.".into(),
                ),
                triqui_size: 10,
            },
        )
        .expect("set job ad mid-run");

    // The ad turned the in-flight prefill run into a real ranking: the
    // button-facing `ranking` flag must be on now (the gui spins the rank
    // buttons on it, not on `running`).
    let start = Instant::now();
    loop {
        let s: libtriqui_client::types::RankState = client
            .call_typed(
                "get_rank_state",
                &StartRankParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("rank state after ad");
        if s.ranking {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "ranking flag never turned on after the ad landed mid-run"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // The same run carries straight through to converged grades.
    let start = Instant::now();
    let final_state = loop {
        let state: libtriqui_client::types::RankState = client
            .call_typed(
                "get_rank_state",
                &StartRankParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("rank state");
        if !state.running {
            break state;
        }
        assert!(
            start.elapsed() < Duration::from_secs(600),
            "the cascade never settled after the mid-run ad"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert!(
        final_state.converged,
        "the mid-run ad must not stop the cascade: reason {:?}",
        final_state.reason
    );
    let page: libtriqui_client::types::CandidatePage = client
        .call_typed(
            "list_candidates",
            &libtriqui_client::ListCandidatesParams {
                job_id: job_id.clone(),
                offset: 0,
                limit: 50,
                query: None,
                sort_by: None,
                sort_dir: Some(false),
            },
        )
        .expect("list candidates");
    assert_eq!(page.items.len(), 2, "both CVs listed");
    for cand in &page.items {
        let stages: Vec<&str> = cand.scores.iter().map(|s| s.question_id.as_str()).collect();
        for stage in ["scout", "ad"] {
            assert!(
                stages.contains(&stage),
                "candidate {} must carry a '{}' grade: {stages:?}",
                cand.id,
                stage
            );
        }
    }

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
}

/// The cancel-then-respawn race: cancelling a running cascade and immediately
/// starting a new one must not let the old runner clobber the new run's
/// `rank_state`. Without a generation guard, the old runner's terminal
/// "cancelled" write lands after the new spawn and the gui sees the new run as
/// cancelled while it works (and further spawns stack on top of it).
///
/// The new run is polled for the clobber signature
/// (`running == false && converged == false`) for its whole lifetime and must
/// end converged with no cancel reason.
///
/// Full-workflow: a real engine, so the cancelled runner is genuinely
/// mid-embed when the cancel+spawn lands — expensive-test pattern.
#[test]
fn a_cascade_spawned_over_a_cancelled_run_keeps_its_state() {
    if std::env::var("IK_SKIP_EXPENSIVE").is_ok() {
        eprintln!("skipping: expensive (real engine, real model cache)");
        return;
    }
    let (serve, client, app_dir) = spawn_serve("respring");
    let cv_root = redirect_cv_root_to_temp(&client, "respring");
    let job_id = create_job_with_ad(
        &client,
        "Re-Spring Engineer",
        "Senior Rust engineer with systems programming experience.",
    );

    // The import auto-starts cascade A; it is mid-scout/mid-wave (engine
    // work, seconds) by the time the next two calls land.
    import_two_txt_cvs(&client, &cv_root, &app_dir, &job_id);

    let rank_state = |job_id: &str| -> libtriqui_client::types::RankState {
        client
            .call_typed(
                "get_rank_state",
                &StartRankParams {
                    job_id: job_id.to_string(),
                },
            )
            .expect("rank state")
    };

    // Cancel A, then immediately spawn B over it. The spawn is allowed
    // precisely because A's run was cancelled.
    client
        .call(
            "cancel_rank",
            &libtriqui_client::CancelRankParams {
                job_id: job_id.clone(),
            },
        )
        .expect("cancel cascade A");
    let _: serde_json::Value = client
        .call_typed(
            "start_rank",
            &StartRankParams {
                job_id: job_id.clone(),
            },
        )
        .expect("spawn cascade B over the cancelled run");

    // B's whole lifetime: the state must always read as a live run (or the
    // converged end state) — never as cancelled-and-stopped, which is A's
    // terminal write bleeding onto B's row.
    let start = Instant::now();
    loop {
        let state = rank_state(&job_id);
        assert!(
            state.running || state.converged,
            "B's rank state was clobbered mid-run: {state:?}"
        );
        if state.converged {
            assert!(
                state.reason.is_none(),
                "B converged but carries a cancel reason: {:?}",
                state.reason
            );
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(600),
            "cascade B never converged"
        );
        std::thread::sleep(Duration::from_millis(250));
    }

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
}

/// A `rank_state` row claiming a run in flight (`complete = 0, cancelled = 0`)
/// is only truthful while the process that spawned the cascade lives. A quit
/// or crash mid-cascade leaves the row behind, and the next launch inherits a
/// job that reports running forever and refuses every new cascade — the
/// wedged-job bug. Serve start must recover the stale row.
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

/// Everything the import of an ad-less job starts must be ad-independent work
/// — and everything ad-dependent must wait for a query. The import-time
/// cascade runs pre-ad and converges having embedded the scout vectors (stage
/// 0.5, whole corpus) and the heavy chunk vectors (the waves); no scout/ad
/// grade may exist yet. When the ad then lands, grades come from the vectors
/// already on disk — no CV is re-embedded.
///
/// Full-workflow: real engine, real model cache (like the other cascade
/// tests), so it follows the expensive-test pattern.
#[test]
fn a_job_imported_without_an_ad_embeds_the_corpus_and_grades_when_the_ad_lands() {
    if std::env::var("IK_SKIP_EXPENSIVE").is_ok() {
        eprintln!("skipping: expensive (real engine, real model cache)");
        return;
    }
    let (serve, client, app_dir) = spawn_serve("pread");
    let cv_root = redirect_cv_root_to_temp(&client, "pread");
    let job: libtriqui_client::types::Job = client
        .call_typed("create_job", &EmptyParams {})
        .expect("create job");
    let job_id = job.id.clone();
    // Title-only update: the ad stays unset (the gui's Setup pane flow before
    // the user pastes an ad).
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Pre-Ad Engineer".into(),
                job_ad_text: None,
                triqui_size: 10,
            },
        )
        .expect("set title");

    import_two_txt_cvs(&client, &cv_root, &app_dir, &job_id);

    // The import auto-started the cascade pre-ad; wait it out. It converges by
    // embedding — the only work it has.
    let rank_state = |job_id: &str| -> libtriqui_client::types::RankState {
        client
            .call_typed(
                "get_rank_state",
                &StartRankParams {
                    job_id: job_id.to_string(),
                },
            )
            .expect("rank state exists once the import cascade started")
    };
    let start = Instant::now();
    let pre_ad = loop {
        let state = rank_state(&job_id);
        if !state.running {
            break state;
        }
        assert!(
            start.elapsed() < Duration::from_secs(600),
            "the pre-ad cascade never settled"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert!(
        pre_ad.converged,
        "the pre-ad run converges (it embeds everyone): reason {:?}",
        pre_ad.reason
    );

    // The right-moment nudge: everything embeddable is embedded, the ad is
    // still missing — job_ui_state now points at the one remaining step
    // (the gui renders this as the dismissible "add the job ad" banner).
    let pre_ad_ui: libtriqui_client::types::JobUiState = client
        .call_typed(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("job_ui_state after pre-ad convergence");
    assert_eq!(
        pre_ad_ui.nudge.as_deref(),
        Some("add-job-ad"),
        "converged ad-less prefill → the add-job-ad nudge must fire"
    );

    // The vectors are on disk — scout AND heavy model — while the grades are
    // not: nothing was scoreable.
    {
        let conn = open_db(&app_dir.join("triqui.db"));
        let models = scalar(
            &conn,
            &format!(
                "SELECT COUNT(DISTINCT vector_model) FROM candidate_chunks
                 WHERE candidate_id IN (SELECT id FROM candidates WHERE job_id = '{job_id}')"
            ),
        );
        assert!(
            models >= 2,
            "pre-ad import must have extracted both scout and heavy vectors, got {models} model(s)"
        );
        let score_rows = scalar(
            &conn,
            &format!(
                "SELECT COUNT(*) FROM candidate_scores WHERE candidate_id IN
                 (SELECT id FROM candidates WHERE job_id = '{job_id}')"
            ),
        );
        assert_eq!(
            score_rows, 0,
            "no stage may write a grade without a query (no ad, no criteria)"
        );
    }

    // The ad lands after the pre-ad run converged: `update_job` spawns the
    // scoring-only cascade.
    let _: libtriqui_client::types::Job = client
        .call_typed(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Pre-Ad Engineer".into(),
                job_ad_text: Some(
                    "Senior Rust engineer with systems programming experience.".into(),
                ),
                triqui_size: 10,
            },
        )
        .expect("set job ad");
    let start = Instant::now();
    let post_ad = loop {
        let state = rank_state(&job_id);
        if !state.running {
            break state;
        }
        assert!(
            start.elapsed() < Duration::from_secs(600),
            "the scoring cascade never settled"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert!(
        post_ad.converged,
        "the ad-landed run converges: reason {:?}",
        post_ad.reason
    );

    // The ad landed — the nudge's premise is gone, it must not linger.
    let post_ad_ui: libtriqui_client::types::JobUiState = client
        .call_typed(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("job_ui_state after the ad landed");
    assert_eq!(
        post_ad_ui.nudge, None,
        "with an ad set there is nothing to nudge about"
    );

    // Grades now, for every CV, from the vectors extracted pre-ad.
    let page: libtriqui_client::types::CandidatePage = client
        .call_typed(
            "list_candidates",
            &libtriqui_client::ListCandidatesParams {
                job_id: job_id.clone(),
                offset: 0,
                limit: 50,
                query: None,
                sort_by: None,
                sort_dir: Some(false),
            },
        )
        .expect("list candidates");
    assert_eq!(page.items.len(), 2, "both CVs listed");
    for cand in &page.items {
        let stages: Vec<&str> = cand.scores.iter().map(|s| s.question_id.as_str()).collect();
        for stage in ["scout", "ad"] {
            assert!(
                stages.contains(&stage),
                "candidate {} must carry a '{}' grade after the ad lands: {stages:?}",
                cand.id,
                stage
            );
        }
    }

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&app_dir);
    let _ = std::fs::remove_dir_all(&cv_root);
}

/// The job ad as a **file**: default path, empty seeding, adoption of edits,
/// watcher pickup, "Change…", and recreation after deletion. Cheap — no CVs are
/// staged, so nothing is embedded and no rank actually runs.
///
/// The watcher arm is the important half: the user edits the ad in their own
/// editor, so the app has to notice the save by itself and mark the job dirty.
/// That is the whole reason the ad is a file rather than app state.
#[test]
fn job_ad_file_is_seeded_watched_and_re_read() {
    let (serve, client, cv_root, _cfg_path) = spawn_serve_isolated_cv_root("jobadfile");
    let events = spawn_event_collector(&client);

    let job_id = create_job_with_ad(
        &client,
        "Field Technician",
        "Diesel mechanic wanted: injectors, turbos and fuel systems.",
    );
    // Start from a job with no ad, so the empty seeding is observable.
    client
        .call_typed::<libtriqui_client::types::Job>(
            "update_job",
            &libtriqui_client::UpdateJobParams {
                job_id: job_id.clone(),
                title: "Field Technician".into(),
                job_ad_text: None,
                triqui_size: 10,
            },
        )
        .expect("clear the ad");

    // -- 1. Asking for the ad resolves the default path and seeds it empty. --
    let ad: libtriqui_client::types::JobAdFile = client
        .call_typed(
            "get_job_ad",
            &libtriqui_client::GetJobAdParams {
                job_id: job_id.clone(),
            },
        )
        .expect("get job ad");
    let ad_path = std::path::PathBuf::from(&ad.path);
    assert!(
        ad_path.starts_with(&cv_root) && ad_path.ends_with("job-ad.txt"),
        "the ad lives beside the CV folder under the cv root: {} (root {})",
        ad.path,
        cv_root.display()
    );
    assert!(ad_path.is_file(), "the default ad file was created");
    assert!(
        std::fs::read_to_string(&ad_path).unwrap().is_empty(),
        "a job with no ad gets an EMPTY file, never invented content"
    );

    // The resolved path is persisted, so the gui shows the same one next time.
    let detail: libtriqui_client::types::JobDetail = client
        .call_typed(
            "get_job_detail",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("detail");
    assert_eq!(detail.job.job_ad_path.as_deref(), Some(ad.path.as_str()));
    assert!(
        detail.job.job_ad_text.is_none(),
        "an empty ad file leaves the job with no ad (never a blank-but-set one)"
    );

    // -- 2. Writing the file, then re-reading it, adopts the text. -----------
    let real_ad = "Diesel mechanic wanted: injectors, turbos and fuel systems. Full benefits.";
    std::fs::write(&ad_path, real_ad).expect("write the ad file");
    let reloaded: libtriqui_client::types::Job = client
        .call_typed(
            "reload_job_ad",
            &libtriqui_client::ReloadJobAdParams {
                job_id: job_id.clone(),
            },
        )
        .expect("reload job ad");
    assert_eq!(reloaded.job_ad_text.as_deref(), Some(real_ad));

    // -- 3. The watcher picks an edit up on its own — no rpc, no button. -----
    //    `watch_job` is what the gui sends when a job is opened.
    client
        .call_typed::<libtriqui_client::types::Job>(
            "watch_job",
            &libtriqui_client::WatchJobParams {
                job_id: job_id.clone(),
            },
        )
        .expect("watch the job");
    let edited = format!("{real_ad} Mention your certification.");
    std::fs::write(&ad_path, &edited).expect("edit the ad in an external editor");

    let start = Instant::now();
    loop {
        let d: libtriqui_client::types::JobDetail = client
            .call_typed(
                "get_job_detail",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("detail while waiting for the watcher");
        if d.job.job_ad_text.as_deref() == Some(edited.as_str()) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the ad file change was never picked up; last ad = {:?}",
            d.job.job_ad_text
        );
        std::thread::sleep(Duration::from_millis(250));
    }
    let ui: libtriqui_client::types::JobUiState = client
        .call_typed(
            "job_ui_state",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("job ui state");
    assert!(
        ui.rerank_dirty,
        "an ad edit is a ranking input, so the job must read as needing a re-rank"
    );

    // -- 4. "Change…" points the job at a different file. -------------------
    let other = cv_root.join("elsewhere").join("ad.md");
    std::fs::create_dir_all(other.parent().unwrap()).expect("mkdir");
    let other_text = "A completely different advertisement, pasted elsewhere.";
    std::fs::write(&other, other_text).expect("write the second ad file");
    let other_str = other.to_string_lossy().to_string();

    let switched: libtriqui_client::types::JobAdFile = client
        .call_typed(
            "set_job_ad_path",
            &libtriqui_client::SetJobAdPathParams {
                job_id: job_id.clone(),
                path: other_str.clone(),
            },
        )
        .expect("change the ad file");
    assert_eq!(
        switched.path, other_str,
        "the chosen path is now the ad path"
    );
    assert_eq!(switched.text, other_text, "the new file's text is adopted");

    let detail: libtriqui_client::types::JobDetail = client
        .call_typed(
            "get_job_detail",
            &GetJobDetailParams {
                job_id: job_id.clone(),
            },
        )
        .expect("detail after change");
    assert_eq!(detail.job.job_ad_path.as_deref(), Some(other_str.as_str()));
    assert_eq!(detail.job.job_ad_text.as_deref(), Some(other_text));

    // -- 4b. A disk edit announces itself, so the pane's read-only mirror of
    //        the file follows the editor. `rank-progress` cannot cover this: on
    //        a job with no extracted CVs no cascade ever starts.
    client
        .call_typed::<libtriqui_client::types::Job>(
            "watch_job",
            &libtriqui_client::WatchJobParams {
                job_id: job_id.clone(),
            },
        )
        .expect("re-watch for the event check");
    let marker = events.lock().expect("event log lock").len();
    std::fs::write(&other, "Announced over the event bus.").expect("edit for the event");
    // Poll only past `marker`: an earlier job-ad-reloaded (emitted by the
    // "Change…" step) is already in the log and a whole-log scan would match it.
    let start = Instant::now();
    loop {
        let seen = events
            .lock()
            .expect("event log lock")
            .iter()
            .skip(marker)
            .any(|(name, data)| name == "job-ad-reloaded" && data["job_id"] == job_id.as_str());
        if seen {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the ad reload event never arrived; the pane's read-only mirror of the \
             file would sit stale after an external edit"
        );
        std::thread::sleep(Duration::from_millis(250));
    }

    // -- 4c. The save shape a real editor produces: write a sibling temp file
    //        and rename it over the target (atomic replace), rather than
    //        truncating the file in place. Many editors do this to avoid
    //        truncating the user's file if the save fails, and a watcher set up
    //        on the file path can miss it entirely.
    let atomic_text = "Saved by an editor that replaces the file atomically.";
    let tmp_sibling = other.with_extension("txt.swp");
    std::fs::write(&tmp_sibling, atomic_text).expect("write the editor's temp file");
    std::fs::rename(&tmp_sibling, &other).expect("atomic replace (editor save)");
    let start = Instant::now();
    loop {
        let d: libtriqui_client::types::JobDetail = client
            .call_typed(
                "get_job_detail",
                &GetJobDetailParams {
                    job_id: job_id.clone(),
                },
            )
            .expect("detail while waiting for the atomic save");
        if d.job.job_ad_text.as_deref() == Some(atomic_text) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "an ATOMIC editor save was never picked up (last ad = {:?}); a watcher on \
             the file path misses the write-temp-then-rename that most editors do",
            d.job.job_ad_text
        );
        std::thread::sleep(Duration::from_millis(250));
    }

    // -- 5. A deleted ad file is recreated empty, not left dangling. --------
    std::fs::remove_file(&other).expect("delete the chosen ad file");
    let recreated: libtriqui_client::types::JobAdFile = client
        .call_typed(
            "get_job_ad",
            &libtriqui_client::GetJobAdParams {
                job_id: job_id.clone(),
            },
        )
        .expect("get job ad after deletion");
    assert!(recreated.created, "a deleted ad file is recreated");
    assert!(
        recreated.text.is_empty(),
        "...and it is empty, not stale text"
    );
    assert!(
        std::path::Path::new(&recreated.path).is_file(),
        "the recreated file exists on disk"
    );

    serve.shutdown();
    let _ = std::fs::remove_dir_all(&cv_root);
}
