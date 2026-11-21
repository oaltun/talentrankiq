use libmodeldownloader::{UrlDownload, download_url, cli::Cli, cli::Artifact, Downloader};
use std::path::PathBuf;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = PathBuf::from("/Users/apple1/Documents/repo/act/ik/activeik/talentrankiq/target/release/model_cache");
    std::fs::create_dir_all(&model_dir)?;

    let artifacts = vec![
        // Harrier 600M ONNX
        ("harrier-oss-v1-0.6b", "ort", "fp16", "model_fp16.onnx", "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_fp16.onnx", false),
        ("harrier-oss-v1-0.6b", "ort", "q4", "model_q4.onnx", "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_q4.onnx", false),
        // Harrier 600M MLX - uses HF source format for repo fetch
        ("harrier-oss-v1-0.6b", "mlx", "4bit", "model.safetensors", "hf:majentik/harrier-oss-v1-0.6b-MLX-4bit", true),
        // Harrier 270M ONNX
        ("harrier-oss-v1-270m", "ort", "q4f16", "model_q4f16.onnx", "https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx", false),
        // Harrier 270M MLX
        ("harrier-oss-v1-270m", "mlx", "4bit", "model.safetensors", "hf:majentik/harrier-oss-v1-270m-MLX-4bit", true),
    ];

    for (slug, backend, quantization, primary_file, url_or_source, is_hf_source) in artifacts {
        println!("=== Downloading {} {} {} ===", slug, backend, quantization);
        
        let path = if is_hf_source {
            // Use HF source mode (CLI mode) for MLX
            let cli = Cli {
                config: None,
                model: Some(slug.to_string()),
                source: Some(url_or_source.to_string()),
                url: None,
                slug: None,
                revision: "main".to_string(),
                model_dir: Some(model_dir.clone()),
                force: false,
                dry_run: false,
                files: vec![],
                extra_files: vec![],
                artifact: if backend == "mlx" { Artifact::Mlx } else { Artifact::Onnx },
                sha256: None,
                verbose: false,
                logfile: None,
            };
            let path = Downloader::run_with_progress(cli, Some(Arc::new(Mutex::new(|ev| {
                match ev {
                    libmodeldownloader::DownloadEvent::Finished { ref file, .. } => {
                        println!("  Downloaded: {}", file);
                    }
                    libmodeldownloader::DownloadEvent::Cached { ref file, bytes } => {
                        println!("  Cached: {} ({} bytes)", file, bytes);
                    }
                    _ => {}
                }
            }))))?;
            path
        } else {
            // Use URL mode for ONNX
            let req = UrlDownload::new(url_or_source, slug, model_dir.clone(), None);
            download_url(req, |ev| {
                match ev {
                    libmodeldownloader::DownloadEvent::Finished { file, .. } => {
                        println!("  Downloaded: {}", file);
                    }
                    libmodeldownloader::DownloadEvent::Cached { file, bytes } => {
                        println!("  Cached: {} ({} bytes)", file, bytes);
                    }
                    _ => {}
                }
            })?
        };

// Find the downloaded file
        let file_path = if backend == "mlx" {
            // CLI mode returns the main directory directly
            path.join(primary_file)
        } else {
            // URL mode returns the slug directory
            path.join("repo").join("onnx").join(primary_file)
        };
        
        if file_path.exists() {
            let bytes = std::fs::read(&file_path)?;
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let sha256 = hex::encode(hasher.finalize());
            println!("  SHA256: {}", sha256);
            
            // Also print fields for easy config.rs update
            println!("  URL/Source: {}", url_or_source);
            println!("  slug: {}", slug);
            println!("  backend: {}", backend);
            println!("  quantization: {}", quantization);
            println!();
        } else {
            eprintln!("  ERROR: File not found: {:?}", file_path);
        }
    }

    Ok(())
}