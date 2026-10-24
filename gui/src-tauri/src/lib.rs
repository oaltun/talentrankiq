//! Tauri application library entry point (thin gui pattern).
//!
//! The gui's Rust layer owns no domain logic and spawns the `triq` CLI
//! as a subprocess. All domain logic, the engine, DB, model management,
//! and CV watching live in the CLI process. The GUI communicates via
//! JSONL over stdin/stdout.

mod bindings;
mod cli;
mod cli_subprocess;
mod commands;
mod triqi_client;
mod utils;

use std::panic;
use tauri::{Manager, RunEvent, WindowEvent};

// The DTO/error vocabulary the command shells speak (and specta exports).
// Re-exported under the historical `crate::types` path.
pub use triqi_client::types;

// Re-export only what's needed externally
pub use triqi_client::types::DEFAULT_QUICK_PANE_SHORTCUT;

/// Application entry point. Sets up all plugins and initializes the app.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // `--cpu-only` (non-default diagnostic flag): pin the engine arm to the
    // ONNX/CPU runtime with slow caps (1 intra thread, batch 4) to simulate a
    // weak machine. Process-global, read once when the engine starts.
    if std::env::args().any(|a| a == "--cpu-only") {
        std::env::set_var("TRIQI_CPU_ONLY", "1");
        log::info!("--cpu-only: forcing ONNX/CPU arm with minimal caps");
    }

    let builder = bindings::generate_bindings();

    // Export TypeScript bindings in debug builds
    #[cfg(debug_assertions)]
    bindings::export_ts_bindings();

    // Build with common plugins
    let mut app_builder = tauri::Builder::default();

    // Single instance plugin must be registered FIRST
    // When user tries to open a second instance, focus the existing window instead
    #[cfg(desktop)]
    {
        app_builder = app_builder.plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
                let _ = window.unminimize();
            }
        }));
    }

    // Window state plugin - saves/restores window position and size
    // Note: quick-pane is denylisted because it's an NSPanel and calling is_maximized() on it crashes
    // See: https://github.com/tauri-apps/plugins-workspace/issues/1546
    #[cfg(desktop)]
    {
        app_builder = app_builder.plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(tauri_plugin_window_state::StateFlags::all())
                .with_denylist(&["quick-pane"])
                .build(),
        );
    }

    // Updater plugin for in-app updates
    #[cfg(desktop)]
    {
        app_builder = app_builder.plugin(tauri_plugin_updater::Builder::new().build());
    }

    app_builder = app_builder
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .plugin({
            #[allow(unused_mut)]
            let mut targets = vec![
                // Always log to stdout for development
                tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                // Log to system logs on macOS (appears in Console.app)
                #[cfg(target_os = "macos")]
                tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir {
                    file_name: None,
                }),
            ];
            // Log to webview console — excluded on Linux where the WebKitGTK webview
            // doesn't exist during setup(), causing app.emit() to deadlock on the IPC socket.
            #[cfg(not(target_os = "linux"))]
            targets.push(tauri_plugin_log::Target::new(
                tauri_plugin_log::TargetKind::Webview,
            ));
            tauri_plugin_log::Builder::new()
                // Use Debug level in development, Info in production
                .level(if cfg!(debug_assertions) {
                    log::LevelFilter::Debug
                } else {
                    log::LevelFilter::Info
                })
                .targets(targets)
                .build()
        });

    // macOS: Add NSPanel plugin for native panel behavior
    #[cfg(target_os = "macos")]
    {
        app_builder = app_builder.plugin(tauri_nspanel::init());
    }

    app_builder
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_persisted_scope::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_os::init())
        .setup(|app| {
            // Install global panic hook early to catch panics in any thread
            let orig_hook = panic::take_hook();
            panic::set_hook(Box::new(move |info| {
                let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                    format!("APP PANIC: {}", s)
                } else if let Some(s) = info.payload().downcast_ref::<String>() {
                    format!("APP PANIC: {}", s)
                } else {
                    "APP PANIC: unknown payload".to_string()
                };
                let loc = info
                    .location()
                    .map(|l| format!(" at {}:{}:{}", l.file(), l.line(), l.column()))
                    .unwrap_or_default();
                let full_msg = format!("{}", msg);
                eprintln!("{}", full_msg);
                eprintln!("  Location:{}", loc);
                eprintln!("  Thread: {:?}", std::thread::current().id());

                // Write to crash log
                let log_path = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
                let log_file = std::path::Path::new(&log_path).join("talentrankiq_app_crash.log");
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_file)
                {
                    use std::io::Write;
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let _ = writeln!(file, "[{}] {}", timestamp, full_msg);
                    let _ = writeln!(file, "  Location:{}", loc);
                    let _ = writeln!(file, "  Thread: {:?}", std::thread::current().id());
                }
                eprintln!("Crash log written to: {:?}", log_file);

                orig_hook(info);
            }));

            log::info!("Application starting up");
            log::debug!(
                "App handle initialized for package: {}",
                app.package_info().name
            );

            // Managed bridge state: the (single) subprocess bridge.
            app.manage(cli::Bridge::new());

            // Config is loaded from bundled triq.yaml (via TRIQ_CONFIG).
            // For development, an optional TRIQ_CONFIG env var can point
            // to a custom config file.
            if let Ok(resource_dir) = app.path().resource_dir() {
                let config_path = resource_dir.join("triq.yaml");
                if config_path.exists() {
                    std::env::set_var("TRIQ_CONFIG", config_path.to_string_lossy().to_string());
                    log::info!("Using bundled config: {}", config_path.display());
                }
            }

            // Set platform-appropriate model cache directory:
            // macOS: ~/Library/Caches/com.talentrankiq.triqui/models
            // Windows: %LOCALAPPDATA%\com.talentrankiq.triqui\models
            // Linux: ~/.cache/talentrankiq/models
            if let Ok(cache_dir) = app.path().app_cache_dir() {
                let model_dir = cache_dir.join("models");
                std::env::set_var("TRIQI_MODEL_DIR", model_dir.to_string_lossy().to_string());
            }

            // Spawn the subprocess bridge now so its CV-dir watchers and the engine
            // status events are live before the frontend asks for anything.
            // The engine itself is NOT started here — the frontend calls
            // `start_runtime` once React has loaded, so launch isn't blocked
            // waiting for model loading.
            let app_dir = app
                .path()
                .app_data_dir()
                .map_err(|e| format!("Failed to get app data directory: {e}"))?;
            std::fs::create_dir_all(&app_dir)
                .map_err(|e| format!("Failed to create app data directory: {e}"))?;
            cli::extract_benchmark_cvs(&app_dir);
            if let Err(e) = cli::spawn_bridge(
                app.handle(),
                &app_dir,
                &app.package_info().version.to_string(),
            ) {
                // Non-fatal: the bridge retries on the first command.
                log::error!("Failed to start subprocess bridge: {e}");
            }

            // Set up global shortcut plugin (without any shortcuts - we register them separately)
            #[cfg(desktop)]
            {
                use tauri_plugin_global_shortcut::Builder;

                app.handle().plugin(Builder::new().build())?;
            }

            // Load saved preferences and register the quick pane shortcut
            #[cfg(desktop)]
            {
                let saved_shortcut = commands::preferences::load_quick_pane_shortcut(app.handle());
                let shortcut_to_register = saved_shortcut
                    .as_deref()
                    .unwrap_or(DEFAULT_QUICK_PANE_SHORTCUT);

                log::info!("Registering quick pane shortcut: {shortcut_to_register}");
                commands::quick_pane::register_quick_pane_shortcut(
                    app.handle(),
                    shortcut_to_register,
                )?;
            }

            // Create the quick pane window (hidden) - must be done on main thread
            if let Err(e) = commands::quick_pane::init_quick_pane(app.handle()) {
                log::error!("Failed to create quick pane: {e}");
                // Non-fatal: app can still run without quick pane
            }

            // NOTE: Application menu is built from JavaScript for i18n support
            // See src/lib/menu.ts for the menu implementation

            Ok(())
        })
        .invoke_handler(builder.invoke_handler())
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| match &event {
            // macOS: Hide the main window instead of quitting so the dock icon can reopen it
            // and the quick-pane shortcut works independently of the main window.
            // On other platforms, the close proceeds normally and the app exits.
            RunEvent::WindowEvent {
                label,
                event: WindowEvent::CloseRequested { api, .. },
                ..
            } if label == "main" => {
                #[cfg(target_os = "macos")]
                {
                    api.prevent_close();

                    // Save window state before hiding
                    use tauri_plugin_window_state::{AppHandleExt, StateFlags};
                    if let Err(e) = app_handle.save_window_state(StateFlags::all()) {
                        log::warn!("Failed to save window state: {e}");
                    }

                    // Hide the window, not the app. app_handle.hide() calls NSApplication.hide()
                    // which sets system-level hidden state — showing an NSPanel while hidden
                    // causes macOS to unhide the entire app, including the main window.
                    if let Some(window) = app_handle.get_webview_window("main") {
                        let _ = window.hide();
                        log::info!("Main window hidden");
                    }
                }
            }

            // macOS: Dock icon clicked — reopen the main window if it was hidden
            #[cfg(target_os = "macos")]
            RunEvent::Reopen { .. } => {
                if let Some(window) = app_handle.get_webview_window("main") {
                    if !window.is_visible().unwrap_or(true) {
                        let _ = window.show();

                        // The window-state plugin only auto-restores on app startup, not after
                        // a hide/show cycle. Without this the window can appear at stale coords.
                        use tauri_plugin_window_state::{StateFlags, WindowExt};
                        let _ = window.restore_state(StateFlags::all());

                        let _ = window.set_focus();
                        log::info!("Main window reopened from dock");
                    }
                }
            }

            // Cleanup on actual exit (Cmd+Q, menu Quit, or window close on non-macOS).
            // RunEvent::Exit fires reliably before the process exits, unlike ExitRequested
            // which doesn't fire for Cmd+Q on macOS (tauri-apps/tauri#9198).
            RunEvent::Exit => {
                log::info!("Application exiting — performing cleanup");

                // Hide the quick-pane panel to prevent crashes during teardown
                #[cfg(target_os = "macos")]
                {
                    use tauri_nspanel::ManagerExt;
                    if let Ok(panel) = app_handle.get_webview_panel("quick-pane") {
                        panel.hide();
                    }
                }

                // Unregister global shortcuts
                #[cfg(desktop)]
                {
                    use tauri_plugin_global_shortcut::GlobalShortcutExt;
                    if let Err(e) = app_handle.global_shortcut().unregister_all() {
                        log::warn!("Failed to unregister global shortcuts: {e}");
                    }
                }

                // Shut the subprocess bridge down (signals EOF; the CLI
                // then tears the engine down itself).
                cli::shutdown_bridge(app_handle);

                log::info!("Cleanup complete");
            }

            _ => {}
        });
}
