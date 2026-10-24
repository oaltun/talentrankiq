// Build script for the Tauri GUI app.
// Copies the `triq` CLI binary into the app bundle so it's available
// at Contents/MacOS/triq with the same sandbox/entitlements as the GUI.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Only run on macOS for the app bundle
    if cfg!(target_os = "macos") {
        // Get the workspace root (where target/release/triq is)
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        let triq_src = workspace_root.join("target/release/triq");

        // Tauri puts the .app in target/release/bundle/macos/
        let out_dir = env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| workspace_root.join("target"));
        let app_bundle = out_dir.join("release/bundle/macos/Triqui.app/Contents/MacOS");
        let triq_dst = app_bundle.join("triq");

        if triq_src.exists() {
            // Create the MacOS directory if needed
            fs::create_dir_all(&app_bundle).unwrap();
            // Copy the binary
            fs::copy(&triq_src, &triq_dst).unwrap();
            println!("cargo:info=Copied triq CLI to {:?}", triq_dst);

            // Make it executable
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&triq_dst).unwrap().permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&triq_dst, perms).unwrap();
            }
        } else {
            println!(
                "cargo:warning=triq binary not found at {:?}, skipping bundle",
                triq_src
            );
        }
    }

    // Tell cargo to rerun if triq binary changes
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join("target/release/triq").display()
    );
}
