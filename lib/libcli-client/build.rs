//! Build script to generate JSON Schema and TypeScript bindings from the
//! specta-annotated types in `libcli-client` and `libcli-client-types`.

use libcli_client_types::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use specta::Type;
use specta::export;
use specta_typescript::Typescript;

// Envelope types (defined inline to avoid circular build dependency)
#[derive(Debug, Clone, Serialize, Deserialize, Type, JsonSchema)]
struct Request {
    id: u64,
    cmd: String,
    #[serde(default)]
    params: Value,
    /// Protocol version sent by the client (e.g., "1.2.0").
    /// The server uses this to verify compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    protocol_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type, JsonSchema)]
struct ProtocolError {
    code: String,
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domain: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type, JsonSchema)]
struct Response {
    id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ok: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    err: Option<ProtocolError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type, JsonSchema)]
struct Event {
    event: String,
    data: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type, JsonSchema)]
#[serde(untagged)]
enum Inbound {
    Response(Response),
    Event(Event),
}

fn main() {
    // Generate JSON Schema using schemars
    let mut root_schema = schemars::schema::RootSchema::default();
    
    // Helper to add types to schema - extracts both root and definitions
    macro_rules! add_type {
        ($($ty:ty),*) => {
            $(
                let settings = schemars::gen::SchemaGenerator::new(schemars::gen::SchemaSettings::draft07());
                let s = settings.into_root_schema_for::<$ty>();
                // Add the root schema as a definition
                let type_name = std::any::type_name::<$ty>();
                let short_name = type_name.split("::").last().unwrap_or(type_name);
                root_schema.definitions.insert(short_name.into(), schemars::schema::Schema::Object(s.schema));
                // Also merge all referenced definitions
                root_schema.definitions.extend(s.definitions);
            )*
        };
    }

    add_type!(
        Request,
        Response,
        Event,
        Inbound,
        ProtocolError,
        DomainError,
        RecoveryError,
        AppPreferences,
        Job,
        JobSummary,
        JobDetail,
        JobAdFile,
        JobUiState,
        SortReport,
        Candidate,
        CandidateDetail,
        CandidateSummary,
        CandidatePage,
        CandidateNeighbors,
        FileType,
        ExtractionStatus,
        Criterion,
        RubricQuestion,
        RubricScore,
        ChunkExplanation,
        QuestionKind,
        GradeSpace,
        AdSource,
        RankState,
        RankProgress,
        SystemBudget,
        SystemLoad,
        ModelKind,
        ModelDownloadProgress,
        ModelDownloadComplete,
        GuiEmbedder,
        TryModelResult,
        TryModelProgress,
        CosinePair,
        EmbedderVariants,
        DownloadStatus,
        VettedEmbedder,
        VettedEmbeddersResponse,
        LicenseStatus,
        ImportProgress,
        EmbeddingProgress,
        ResultsCsvSummary,
        BundleExportOptions,
        BundleExportSummary,
        RuntimeStatusInfo,
        Activity,
        ChunkMatchTarget,
        ChunkSim,
        ChunkMatchResult
    );

    // Write JSON Schema
    std::fs::write(
        "protocol-schema.json",
        serde_json::to_string_pretty(&root_schema).unwrap(),
    )
    .expect("Failed to write protocol-schema.json");

    // TypeScript
    let mut types = export();
    types
        .register::<Request>()
        .register::<Response>()
        .register::<Event>()
        .register::<Inbound>()
        .register::<ProtocolError>()
        .register::<DomainError>()
        .register::<RecoveryError>()
        .register::<AppPreferences>()
        .register::<Job>()
        .register::<JobSummary>()
        .register::<JobDetail>()
        .register::<JobAdFile>()
        .register::<JobUiState>()
        .register::<SortReport>()
        .register::<Candidate>()
        .register::<CandidateDetail>()
        .register::<CandidateSummary>()
        .register::<CandidatePage>()
        .register::<CandidateNeighbors>()
        .register::<FileType>()
        .register::<ExtractionStatus>()
        .register::<Criterion>()
        .register::<RubricQuestion>()
        .register::<RubricScore>()
        .register::<ChunkExplanation>()
        .register::<QuestionKind>()
        .register::<GradeSpace>()
        .register::<AdSource>()
        .register::<RankState>()
        .register::<RankProgress>()
        .register::<SystemBudget>()
        .register::<SystemLoad>()
        .register::<ModelKind>()
        .register::<ModelDownloadProgress>()
        .register::<ModelDownloadComplete>()
        .register::<GuiEmbedder>()
        .register::<TryModelResult>()
        .register::<TryModelProgress>()
        .register::<CosinePair>()
        .register::<EmbedderVariants>()
        .register::<DownloadStatus>()
        .register::<VettedEmbedder>()
        .register::<VettedEmbeddersResponse>()
        .register::<LicenseStatus>()
        .register::<ImportProgress>()
        .register::<EmbeddingProgress>()
        .register::<ResultsCsvSummary>()
        .register::<BundleExportOptions>()
        .register::<BundleExportSummary>()
        .register::<RuntimeStatusInfo>()
        .register::<Activity>()
        .register::<ChunkMatchTarget>()
        .register::<ChunkSim>()
        .register::<ChunkMatchResult>();

    specta_typescript::Typescript::default()
        .bigint(specta_typescript::BigIntExportBehavior::Number)
        .header("/* Auto-generated CLI protocol types. DO NOT EDIT. */\n\n")
        .export_to("protocol.ts", &types)
        .expect("Failed to export TypeScript");

    // Tell cargo to re-run if any source files change
    println!("cargo:rerun-if-changed=../libcli-client-types/src/lib.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/protocol.rs");
    println!("cargo:rerun-if-changed=src/params.rs");
    println!("cargo:rerun-if-changed=src/types.rs");
    println!("cargo:rerun-if-changed=src/client.rs");
    println!("cargo:rerun-if-changed=src/transport.rs");
}