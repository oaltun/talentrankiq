//! In-process engine wiring: the exact `EngineRuntime::start` chain (registry
//! from `triq.yaml` → `libmodelcaller::run_stdio_over` on a thread over an
//! in-memory duplex → `ModelCallerClient::over`). The ORT wrapper runs
//! in-process as well (a dedicated wrapper thread inside modelcaller).
//! Load hits the real ONNX model — skipped under `IK_SKIP_EXPENSIVE`
//! (AGENTS.md convention).

use libmodelcallerclient::ModelCallerClient;
use libmodelcaller::{Registry, StdioArgs, run_stdio_over};

/// Both tests below bring up a full in-process engine (modelcaller loop +
/// in-process onnxwrapperrs wrapper + ONNX model). cargo runs a binary's
/// tests in parallel, and two concurrent engines thrash this low-RAM machine —
/// serialize them.
static ENGINE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn inproc_client() -> (ModelCallerClient, std::thread::JoinHandle<()>) {
    let _ = env_logger::try_init();
    let cfg = libtriqui_core::app_config::load_triq();
    let registry = Registry::from_config(&cfg);
    let args = StdioArgs {
        model_dir: Some(libtriqui_core::app_config::model_dir()),
        revision: cfg.revision.clone(),
        ..Default::default()
    };
    // Side a = host client, side b = the modelcaller loop (a_write → b_read
    // carries requests, b_write → a_read carries responses).
    let (a_write, a_read, b_write, b_read) = libtriqui_client::transport::duplex();
    let handle = std::thread::spawn(move || {
        run_stdio_over(
            registry, args, cfg.engine, cfg.syscap, false, None, b_read, b_write,
        )
        .expect("modelcaller loop")
    });
    let client = ModelCallerClient::over(Box::new(a_read), Box::new(a_write));
    (client, handle)
}

#[test]
fn ping_over_inproc_modelcaller() {
    let _engine = ENGINE_LOCK.lock().unwrap();
    let (client, handle) = inproc_client();
    assert!(client.ping().expect("ping"), "ping must report ok");
    client.shutdown();
    handle.join().expect("loop thread exits cleanly");
}

#[test]
fn load_default_embedder_over_inproc_modelcaller() {
    if std::env::var("IK_SKIP_EXPENSIVE").is_ok() {
        eprintln!("skipping: IK_SKIP_EXPENSIVE set");
        return;
    }
    let _engine = ENGINE_LOCK.lock().unwrap();
    let (client, handle) = inproc_client();
    assert!(client.ping().expect("ping"));
    let model = std::env::var("TRIQI_EMBED_MODEL")
        .unwrap_or_else(|_| "harrier-oss-v1-0.6b-ONNX-q4".to_string());
    client.load(&model).expect("load default embedder");
    client.shutdown();
    handle.join().expect("loop thread exits cleanly");
}

/// The scout runs **alone**: it loads, embeds, and unloads again before the
/// full embedder loads — one resident model at a time (the scout is fast, so
/// reloading it next pass is far cheaper than keeping a second session warm;
/// progressive-ranking design §7).
///
/// This is the test that keeps the scout swap from silently breaking: a new
/// scout slug is a config change, and the only thing that catches a slug whose
/// repo layout the downloader cannot serve (no root `config.json`, tokenizer in
/// a module dir) is actually loading it. It also pins the properties the
/// cascade depends on: the scout produces 1024-d vectors in its own space, it
/// is cross-lingual enough to rank a Turkish CV against an English ad above an
/// unrelated one, and it frees its session so the heavy model can load.
#[test]
fn scout_loads_embeds_unloads_then_the_full_embedder_loads() {
    if std::env::var("IK_SKIP_EXPENSIVE").is_ok() {
        eprintln!("skipping: IK_SKIP_EXPENSIVE set");
        return;
    }
    let _engine = ENGINE_LOCK.lock().unwrap();
    let (client, handle) = inproc_client();
    assert!(client.ping().expect("ping"));

    let scout = std::env::var("TRIQI_SCOUT_MODEL")
        .unwrap_or_else(|_| "static-similarity-mrl-multilingual-v1".to_string());

    // The scout's pass: it is the only model resident.
    client.load(&scout).expect("load scout");

    let embed = |text: &str| -> Vec<f32> {
        let mut vecs = client
            .embed(&scout, &[text], None)
            .expect("scout embed request");
        assert_eq!(vecs.len(), 1, "one vector per text");
        let v: Vec<f32> = vecs.remove(0).into_iter().map(|x| x as f32).collect();
        assert_eq!(v.len(), 1024, "scout must be 1024-d");
        assert!(v.iter().any(|&x| x != 0.0), "vector must not be all-zero");
        // Unit-normalized (the wrapper normalizes by default).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "vector not normalized: |v|={norm}");
        v
    };

    let ad = embed("We are hiring a backend engineer with Rust and PostgreSQL experience.");
    let tr_match = embed(
        "Rust ve PostgreSQL tecrübesi olan bir backend mühendisi arıyoruz. \
         İlan detayları için başvurun.",
    );
    let unrelated = embed("Pastry chef with a focus on laminated dough and viennoiserie.");

    let cosine = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    };
    // The whole reason the scout exists: a Turkish CV must not starve behind an
    // English ad just for its language. If this ever fails, the scout is no
    // longer a fair gate and the cascade's K1 selection is unfair again.
    let match_sim = cosine(&ad, &tr_match);
    let unrelated_sim = cosine(&ad, &unrelated);
    assert!(
        match_sim > unrelated_sim,
        "scout is not cross-lingual enough to gate on: tr_match={match_sim}, unrelated={unrelated_sim}"
    );

    // Pass done: the scout frees its session, then the heavy model loads.
    client.unload(&scout).expect("unload scout after its pass");
    let full = "harrier-oss-v1-0.6b-ONNX-q4";
    client.load(full).expect("load full embedder after the scout unloaded");
    let mut vecs = client
        .embed(full, &["Rust systems engineer."], None)
        .expect("full embedder works after the scout unloaded");
    assert_eq!(vecs.len(), 1);
    assert!(!vecs.remove(0).is_empty(), "full embedder produces vectors");

    client.shutdown();
    handle.join().expect("loop thread exits cleanly");
}
