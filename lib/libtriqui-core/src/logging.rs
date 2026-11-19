//! Logging infrastructure: a `tracing` subscriber whose writer routes events to
//! per-request log files, plus the resolved log-directory precedence.
//!
//! Every RPC handled by the serve loop gets its own timestamped file in the
//! resolved log directory (see [`open_request_log`]). Events are routed through
//! a thread-local set on the thread handling the request, so the `tracing` fmt
//! layer — installed once per process via [`init`] — writes into the right file
//! without per-event plumbing. Outside a request context the writer falls back
//! to stderr.

use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

/// Prefix for per-request log file names: `triq__<YYYY-MM-DD>__<HH-MM-SS-mmm>__<request>.log`.
pub const LOG_FILE_PREFIX: &str = "triq__";

thread_local! {
    /// The file for the request currently being handled on this thread.
    static CURRENT_REQUEST_LOG: RefCell<Option<Arc<Mutex<File>>>> = RefCell::new(None);
}

/// Set the per-request log file for the current thread (called on entry, cleared on exit).
pub fn set_current_request_log(file: Arc<Mutex<File>>) {
    CURRENT_REQUEST_LOG.with(|c| *c.borrow_mut() = Some(file));
}

/// Clear the per-request log file for the current thread (called on exit).
pub fn clear_current_request_log() {
    CURRENT_REQUEST_LOG.with(|c| *c.borrow_mut() = None);
}

/// The per-request log file for the current thread, if any.
pub fn current_request_log() -> Option<Arc<Mutex<File>>> {
    CURRENT_REQUEST_LOG.with(|c| c.borrow().clone())
}

/// Open (create/truncate) the per-request log file for `request` in `log_dir`.
/// Returns `None` when no log directory is resolved or the file cannot be created.
pub fn open_request_log(log_dir: &Path, request: &str) -> Option<Arc<Mutex<File>>> {
    if log_dir.as_os_str().is_empty() {
        return None;
    }
    if let Err(e) = std::fs::create_dir_all(log_dir) {
        tracing::warn!("cannot create log dir {}: {e}", log_dir.display());
        return None;
    }
    let now = chrono::Local::now();
    let millis = format!("{:03}", now.timestamp_micros() % 1000);
    let file_name = format!(
        "{LOG_FILE_PREFIX}{date}__{time}__{req}.log",
        date = now.format("%Y-%m-%d"),
        time = format!("{}-{}", now.format("%H-%M-%S"), millis),
        req = sanitize_request(request),
    );
    let path = log_dir.join(file_name);
    let file = File::create(&path)
        .map_err(|e| {
            tracing::warn!("cannot create request log file {}: {e}", path.display());
            e
        })
        .ok()?;
    Some(Arc::new(Mutex::new(file)))
}

/// Sanitize a request/command name for use in a file name.
fn sanitize_request(request: &str) -> String {
    let cleaned: String = request
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let cleaned = cleaned.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "request".to_string()
    } else {
        cleaned
    }
}

/// The `tracing` fmt writer: routes events to the current thread's request log
/// file, falling back to stderr outside a request context.
pub struct RequestLogWriter;

struct FileWriter(Arc<Mutex<File>>);

impl io::Write for FileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            .write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0
            .lock()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            .flush()
    }
}

impl<'a> MakeWriter<'a> for RequestLogWriter {
    type Writer = Box<dyn io::Write + Send + Sync + 'static>;
    fn make_writer(&'a self) -> Self::Writer {
        CURRENT_REQUEST_LOG.with(|c| {
            let fallback: Box<dyn io::Write + Send + Sync + 'static> = Box::new(io::stderr());
            match c.borrow().as_ref() {
                Some(file) => {
                    let w = FileWriter(Arc::clone(file));
                    Box::new(w) as Box<dyn io::Write + Send + Sync + 'static>
                }
                None => fallback,
            }
        })
    }
}

/// Install the global tracing subscriber: stderr fallback (per RUST_LOG, default
/// `info`) + per-request file routing via the thread-local writer.
pub fn init() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(RequestLogWriter)
        .with_target(false)
        .with_level(true)
        .with_ansi(false)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// Resolve the log directory with the precedence: CLI flag override >
/// `TRIQ_LOG_DIR` env > saved `AppPreferences.log_dir` > OS default.
pub fn resolve_log_dir(app_dir: Option<&Path>, cli_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = cli_override {
        if !dir.as_os_str().is_empty() {
            return Some(dir.to_path_buf());
        }
    }
    if let Ok(dir) = std::env::var("TRIQ_LOG_DIR") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    let app_dir = app_dir?;
    crate::app_config::log_dir_for(app_dir)
}
