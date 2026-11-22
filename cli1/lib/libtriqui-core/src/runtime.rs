//! In-process embedding engine runtime backed by `libmodelcallerclient`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libmodelcallerclient::ModelCallerClient;
use libtriqui_client::types::RuntimeStatusInfo;

pub struct EngineRuntime {
    pub db_path: PathBuf,
    pub status_hook: std::sync::Mutex<Option<Arc<dyn Fn(&RuntimeStatusInfo) + Send + Sync>>>,
    client: Mutex<Option<ModelCallerClient>>,
    active_model: Mutex<String>,
}

pub struct RuntimeConfig {
    model: Option<String>,
    cfg: Option<libconfigreader::Config>,
}

impl RuntimeConfig {
    pub fn discover(model: Option<String>, cfg: Option<libconfigreader::Config>) -> RuntimeConfig {
        RuntimeConfig { model, cfg }
    }
}

struct ClientEmbedder {
    client: ModelCallerClient,
    model: String,
}

impl crate::embed::Embedder for ClientEmbedder {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<f32>, String> {
        let roles: Vec<&str> = texts.iter().map(|_| "document").collect();
        let vecs = self
            .client
            .embed(&self.model, texts, Some(&roles))
            .map_err(|e| e.to_string())?;
        Ok(vecs
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as f32)
            .collect())
    }
}

pub trait ModelClient: Send + Sync {
    fn unload(&self, model: &str) -> Result<(), String>;
    fn load(&self, model: &str) -> Result<(), String>;
    fn switch_model(&self, model: String) -> Result<(), String>;
    fn embed(&self, model: &str, texts: &[&str], opts: Option<&str>) -> Result<Vec<Vec<f32>>, String>;
}

struct RuntimeModelClient {
    client: ModelCallerClient,
}

impl ModelClient for RuntimeModelClient {
    fn unload(&self, model: &str) -> Result<(), String> {
        self.client.unload(model).map_err(|e| e.to_string())
    }

    fn load(&self, model: &str) -> Result<(), String> {
        self.client.load(model).map_err(|e| e.to_string())
    }

    fn switch_model(&self, model: String) -> Result<(), String> {
        self.client
            .model_metadata(&model)
            .map_err(|e| format!("unknown model '{model}': {e}"))?;
        self.client.load(&model).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn embed(&self, model: &str, texts: &[&str], _opts: Option<&str>) -> Result<Vec<Vec<f32>>, String> {
        let roles: Vec<&str> = texts.iter().map(|_| "document").collect();
        let vecs = self
            .client
            .embed(model, texts, Some(&roles))
            .map_err(|e| e.to_string())?;
        Ok(vecs.into_iter().map(|v| v.into_iter().map(|x| x as f32).collect()).collect())
    }
}

impl EngineRuntime {
    pub fn new(cfg: RuntimeConfig, db_path: PathBuf) -> Arc<EngineRuntime> {
        let active_model = cfg
            .model
            .clone()
            .or_else(|| {
                cfg.cfg
                    .as_ref()
                    .and_then(|c| {
                        c.gui_embedders
                            .iter()
                            .find(|ge| ge.default)
                            .map(|ge| ge.slug.clone())
                    })
            })
            .unwrap_or_default();

        let client = ModelCallerClient::spawn_inprocess(std::path::Path::new("triq.yaml"), false, None).ok();

        Arc::new(EngineRuntime {
            db_path,
            status_hook: std::sync::Mutex::new(None),
            client: Mutex::new(client),
            active_model: Mutex::new(active_model),
        })
    }

    pub fn set_status_hook(&self, hook: Arc<dyn Fn(&RuntimeStatusInfo) + Send + Sync>) {
        *self.status_hook.lock().unwrap() = Some(hook);
    }

    pub fn start(&self) {}

    pub fn wait_until_ready(&self, _timeout: Duration) -> Result<(), String> {
        if let Some(client) = self.client.lock().unwrap().as_ref() {
            client.ping().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn shutdown(&self) {
        if let Some(client) = self.client.lock().unwrap().take() {
            client.shutdown();
        }
    }

    pub fn spawn_idle_monitor(&self) {}

    pub fn status_info(&self) -> RuntimeStatusInfo {
        let model = self.active_model.lock().unwrap().clone();
        let has_client = self.client.lock().unwrap().is_some();
        RuntimeStatusInfo {
            status: if has_client { "Running".to_string() } else { "Stopped".to_string() },
            model: if model.is_empty() { None } else { Some(model) },
            dim: 0,
            busy: false,
            error: None,
            backend: None,
            activity: None,
        }
    }

    pub fn open_conn(&self) -> Result<rusqlite::Connection, String> {
        crate::db::open_single_connection_at(&self.db_path)
    }

    pub fn embedder(&self) -> Result<Arc<dyn crate::embed::Embedder>, String> {
        let client = self
            .client
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "no modelcaller client configured".to_string())?;
        let model = self.active_model.lock().unwrap().clone();
        if model.is_empty() {
            return Err("no active model".to_string());
        }
        Ok(Arc::new(ClientEmbedder { client, model }))
    }

    pub fn embedder_for(&self, model: &str) -> Result<Arc<dyn crate::embed::Embedder>, String> {
        let client = self
            .client
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "no modelcaller client configured".to_string())?;
        client
            .model_metadata(model)
            .map_err(|e| format!("unknown model '{model}': {e}"))?;
        Ok(Arc::new(ClientEmbedder {
            client,
            model: model.to_string(),
        }))
    }

    pub fn client(&self) -> Result<Arc<dyn ModelClient>, String> {
        let client = self
            .client
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "no modelcaller client configured".to_string())?;
        Ok(Arc::new(RuntimeModelClient { client }))
    }

    pub fn switch_model(&self, model: String) -> Result<(), String> {
        let client = self
            .client
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "no modelcaller client configured".to_string())?;
        client
            .model_metadata(&model)
            .map_err(|e| format!("unknown model '{model}': {e}"))?;
        client.load(&model).map_err(|e| e.to_string())?;
        *self.active_model.lock().unwrap() = model;
        Ok(())
    }
}

