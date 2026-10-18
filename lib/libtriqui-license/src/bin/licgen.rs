use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, Subcommand};
use ed25519_dalek::{SigningKey, Signer, Verifier};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

const LICENSE_FORMAT: u32 = 1;

#[derive(Parser)]
#[command(name = "licgen", version, about = "TalentRankIQ License Generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new license file
    Generate {
        /// Customer name/identifier
        #[arg(long)]
        customer: String,
        /// License expiration date (YYYY-MM-DD)
        #[arg(long)]
        expires: String,
        /// Comma-separated artifact IDs to authorize (empty = all)
        #[arg(long, default_value = "*")]
        artifacts: String,
        /// Output file path
        #[arg(long, default_value = "license.json")]
        output: PathBuf,
        /// Private key file (PEM format)
        #[arg(long)]
        key: PathBuf,
    },
    /// Verify a license file
    Verify {
        /// License file path
        #[arg(long)]
        license: PathBuf,
        /// Public key file (PEM format)
        #[arg(long)]
        pubkey: PathBuf,
    },
    /// Generate a new Ed25519 key pair
    Keygen {
        /// Output directory for keys
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct License {
    format: u32,
    feature: String,
    customer: String,
    issued_at: u64,
    expires_at: String,
    artifacts: Vec<String>,
    signature: String,
}

#[derive(Debug, Serialize)]
struct SignedPayload {
    format: u32,
    feature: String,
    customer: String,
    issued_at: u64,
    expires_at: String,
    artifacts: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { customer, expires, artifacts, output, key } => {
            generate_license(customer, expires, artifacts, output, key)?
        }
        Commands::Verify { license, pubkey } => verify_license(license, pubkey)?,
        Commands::Keygen { out_dir } => generate_keypair(out_dir)?,
    }
    Ok(())
}

fn generate_license(
    customer: String,
    expires: String,
    artifacts: String,
    output: PathBuf,
    key_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let key_pem = std::fs::read_to_string(&key_path)?;
    let signing_key = parse_private_key(&key_pem)?;

    let artifacts_list: Vec<String> = if artifacts == "*" {
        vec!["*".to_string()]
    } else {
        artifacts.split(',').map(|s| s.trim().to_string()).collect()
    };

    let issued_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let payload = SignedPayload {
        format: LICENSE_FORMAT,
        feature: "judge".to_string(),
        customer: customer.clone(),
        issued_at,
        expires_at: expires.clone(),
        artifacts: artifacts_list.clone(),
    };

    let canonical = serde_json::to_vec(&payload)?;
    let signature = signing_key.sign(&canonical);
    let signature_b64 = BASE64.encode(signature.to_bytes());

    let license = License {
        format: LICENSE_FORMAT,
        feature: "judge".to_string(),
        customer,
        issued_at,
        expires_at: expires,
        artifacts: artifacts_list,
        signature: signature_b64,
    };

    let json = serde_json::to_string_pretty(&license)?;
    std::fs::write(&output, json)?;
    println!("License generated: {}", output.display());
    println!("Customer: {}", license.customer);
    println!("Expires: {}", license.expires_at);
    println!("Artifacts: {:?}", license.artifacts);
    Ok(())
}

fn verify_license(
    license_path: PathBuf,
    pubkey_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let license_content = std::fs::read_to_string(&license_path)?;
    let license: License = serde_json::from_str(&license_content)?;

    let pubkey_pem = std::fs::read_to_string(&pubkey_path)?;
    let verifying_key = parse_public_key(&pubkey_pem)?;

    let payload = SignedPayload {
        format: license.format,
        feature: license.feature.clone(),
        customer: license.customer.clone(),
        issued_at: license.issued_at,
        expires_at: license.expires_at.clone(),
        artifacts: license.artifacts.clone(),
    };
    let canonical = serde_json::to_vec(&payload)?;

    let sig_bytes = BASE64.decode(&license.signature)?;
    let signature = ed25519_dalek::Signature::try_from(sig_bytes.as_slice())?;

    Verifier::verify(&verifying_key, &canonical, &signature)?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let expires = chrono::DateTime::parse_from_rfc3339(&license.expires_at)?.timestamp() as u64;

    if now > expires {
        eprintln!("License EXPIRED (expired at {})", license.expires_at);
        return Err("License expired".into());
    }

    println!("License VALID");
    println!("Customer: {}", license.customer);
    println!("Expires: {}", license.expires_at);
    println!("Artifacts: {:?}", license.artifacts);
    Ok(())
}

fn generate_keypair(out_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut csprng = OsRng;
    let mut secret_bytes = [0u8; 32];
    csprng.fill_bytes(&mut secret_bytes);
    let signing_key = SigningKey::from_bytes(&secret_bytes);
    let verifying_key = signing_key.verifying_key();

    let private_pem = format!(
        "-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----",
        BASE64.encode(signing_key.to_bytes())
    );
    let public_pem = format!(
        "-----BEGIN PUBLIC KEY-----\n{}\n-----END PUBLIC KEY-----",
        BASE64.encode(verifying_key.to_bytes())
    );

    std::fs::write(out_dir.join("private_key.pem"), private_pem)?;
    std::fs::write(out_dir.join("public_key.pem"), public_pem)?;
    std::fs::write(out_dir.join("public_key.raw"), verifying_key.to_bytes())?;

    println!("Keys generated in: {}", out_dir.display());
    println!("  private_key.pem  (keep secret!)");
    println!("  public_key.pem   (embed in app)");
    println!("  public_key.raw   (raw 32 bytes for include_bytes!)");
    Ok(())
}

fn parse_private_key(pem: &str) -> Result<SigningKey, Box<dyn std::error::Error>> {
    let pem = pem.trim();
    let b64 = pem
        .strip_prefix("-----BEGIN PRIVATE KEY-----")
        .and_then(|s| s.strip_suffix("-----END PRIVATE KEY-----"))
        .or_else(|| pem.strip_prefix("-----BEGIN ED25519 PRIVATE KEY-----")
            .and_then(|s| s.strip_suffix("-----END ED25519 PRIVATE KEY-----")))
        .ok_or("Invalid PEM format")?
        .trim();
    let bytes = BASE64.decode(b64)?;
    let key_bytes: [u8; 32] = bytes.as_slice().try_into()?;
    Ok(SigningKey::from_bytes(&key_bytes))
}

fn parse_public_key(pem: &str) -> Result<ed25519_dalek::VerifyingKey, Box<dyn std::error::Error>> {
    let pem = pem.trim();
    let b64 = pem
        .strip_prefix("-----BEGIN PUBLIC KEY-----")
        .and_then(|s| s.strip_suffix("-----END PUBLIC KEY-----"))
        .ok_or("Invalid PEM format")?
        .trim();
    let bytes = BASE64.decode(b64)?;
    let key_bytes: [u8; 32] = bytes.as_slice().try_into()?;
    Ok(ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)?)
}