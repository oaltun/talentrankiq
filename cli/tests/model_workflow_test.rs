use libmodelstorage;
use libcli_client::client::TriqiClient;
use libcli_client::protocol::Event;
use libcli_client::types::{
    GuiEmbedder, ModelDownloadComplete, ModelDownloadProgress, TryModelResult,
};
use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const EMBEDDER_SLUGS: &[&str] = &[
    "harrier-oss-v1-270m-ONNX-q4f16",
    "harrier-oss-v1-270m-MLX-4bit",
    "harrier-oss-v1-0.6b-ONNX-q4",
    "harrier-oss-v1-0.6b-MLX-4bit",
];

fn should_test_embedder(_slug: &str) -> bool {
    true
}

struct TestProcess {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
}

impl TestProcess {
    fn spawn(app_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let binary = manifest_dir.join("../target/debug/triq");
        let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);
        
        let mut cmd = Command::new(&binary);
        cmd.arg("serve");
        cmd.env("TRIQ_APP_DIR", app_dir);
        cmd.env("RUST_LOG", "info");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let process = Self {
            child,
            stdin,
            stdout,
            stderr,
        };

        thread::sleep(Duration::from_secs(2));
        Ok(process)
    }

    fn wait_for_ready(&mut self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let stderr = self.stderr.as_mut().unwrap();
        let reader = BufReader::new(stderr);
        let start = std::time::Instant::now();
        for line in reader.lines() {
            if start.elapsed() > timeout {
                return Err("Timeout waiting for ready".into());
            }
            if let Ok(line) = line {
                if line.contains("serve: ready") {
                    return Ok(());
                }
            }
        }
        Err("Process exited before ready".into())
    }

    fn take_stdio(&mut self) -> (Box<dyn std::io::Read + Send>, Box<dyn std::io::Write + Send>) {
        let stdout = self.stdout.take().unwrap();
        let stdin = self.stdin.take().unwrap();
        (Box::new(stdout), Box::new(stdin))
    }

    fn shutdown(mut self) -> Result<(), Box<dyn std::error::Error>> {
        drop(self.stdin.take());
        let _ = self.child.wait()?;
        Ok(())
    }
}

struct TestClient {
    client: TriqiClient,
    events: Arc<Mutex<Vec<Event>>>,
    _event_thread: thread::JoinHandle<()>,
}

impl TestClient {
    fn new(read: Box<dyn std::io::Read + Send>, write: Box<dyn std::io::Write + Send>) -> Self {
        let client = TriqiClient::over(read, write);
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let event_rx = client.take_event_receiver().expect("event receiver");
        let _event_thread = thread::spawn(move || {
            while let Ok(event) = event_rx.recv() {
                events_clone.lock().unwrap().push(event);
            }
        });
        Self {
            client,
            events,
            _event_thread,
        }
    }

    fn call<T: serde::de::DeserializeOwned>(&self, cmd: &str, params: &impl serde::Serialize) -> Result<T, Box<dyn std::error::Error>> {
        Ok(self.client.call_typed(cmd, params)?)
    }

    fn call_raw(&self, cmd: &str, params: &impl serde::Serialize) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(self.client.call(cmd, params)?)
    }

    fn wait_for_event<F>(&self, predicate: F, timeout: Duration) -> Result<Event, Box<dyn std::error::Error>>
    where
        F: Fn(&Event) -> bool,
    {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err("Timeout waiting for event".into());
            }
            let events = self.events.lock().unwrap();
            if let Some(event) = events.iter().find(|e| predicate(e)) {
                return Ok(event.clone());
            }
            drop(events);
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn wait_for_download_progress(&self, slug: &str, timeout: Duration) -> Result<ModelDownloadProgress, Box<dyn std::error::Error>> {
        let event = self.wait_for_event(
            |e| e.event == "model-download-progress" && e.data["repo_id"].as_str() == Some(slug),
            timeout,
        )?;
        Ok(serde_json::from_value(event.data)?)
    }

    fn wait_for_download_complete(&self, slug: &str, timeout: Duration) -> Result<ModelDownloadComplete, Box<dyn std::error::Error>> {
        let event = self.wait_for_event(
            |e| e.event == "model-download-complete" && e.data["slug"].as_str() == Some(slug),
            timeout,
        )?;
        Ok(serde_json::from_value(event.data)?)
    }
}

fn copy_benchmark_cvs(src: &std::path::Path, dst: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "txt") {
            let dest = dst.join(path.file_name().unwrap());
            std::fs::copy(&path, &dest)?;
        }
    }
    Ok(())
}

fn list_embedders(client: &TestClient) -> Result<Vec<GuiEmbedder>, Box<dyn std::error::Error>> {
    let result: Vec<GuiEmbedder> = client.call("list_embedders", &libcli_client::params::EmptyParams {})?;
    Ok(result)
}

fn find_embedder<'a>(embedders: &'a [GuiEmbedder], slug: &str) -> Option<&'a GuiEmbedder> {
    embedders.iter().find(|e| e.slug == slug)
}

