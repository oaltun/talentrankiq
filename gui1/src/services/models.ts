// Frontend service for model management.

import type { UnlistenFn } from "@tauri-apps/api/event"
import { appInvoke, isTauri } from "../lib/invoke"
import { listenEvent } from "../lib/events"
import type {
  GuiEmbedder,
  DownloadStatus,
  ModelKind,
  ModelDownloadProgress,
  ModelDownloadComplete,
  ScoutModel,
  ComparatorArtifact,
} from "../types"

export type {
  GuiEmbedder,
  DownloadStatus,
  ModelKind,
  ModelDownloadProgress,
  ModelDownloadComplete,
  ScoutModel,
  ComparatorArtifact,
}

export const BROWSER_SCOUT: ScoutModel = {
  downloaded: false,
  name: "static-similarity-mrl-multilingual-v1",
  size_bytes: 0,
  slug: "static-similarity-mrl-multilingual-v1",
}

export const BROWSER_EMBEDDERS: GuiEmbedder[] = [
  {
    slug: "harrier-oss-v1-270m-ONNX-q4f16",
    name: "Harrier 270M ONNX",
    memory_mb: 600,
    downloaded: false,
    dim: 640,
    max_context_window: 32768,
    size_bytes: 0,
    source: "hf:unknown",
    is_default: false,
    comparator: {
      embedder_slug: "harrier-oss-v1-270m-ONNX-q4f16",
      entitled: true,
      exists: false,
      path: "comparator-harrier-270m-v1.json",
    },
  },
  {
    slug: "harrier-oss-v1-270m-MLX-4bit",
    name: "Harrier 270M MLX (Metal)",
    memory_mb: 600,
    downloaded: false,
    dim: 640,
    max_context_window: 32768,
    size_bytes: 0,
    source: "hf:unknown",
    is_default: false,
    comparator: {
      embedder_slug: "harrier-oss-v1-270m-MLX-4bit",
      entitled: true,
      exists: false,
      path: "comparator-harrier-270m-v1.json",
    },
  },
  {
    slug: "harrier-oss-v1-0.6b-ONNX-q4",
    name: "Harrier 600M ONNX",
    memory_mb: 2200,
    downloaded: false,
    dim: 1024,
    max_context_window: 32768,
    size_bytes: 0,
    source: "hf:unknown",
    is_default: false,
    comparator: {
      embedder_slug: "harrier-oss-v1-0.6b-ONNX-q4",
      entitled: true,
      exists: false,
      path: "comparator-harrier-600m-v1.json",
    },
  },
  {
    slug: "harrier-oss-v1-0.6b-MLX-4bit",
    name: "Harrier 600M MLX (Metal)",
    memory_mb: 2200,
    downloaded: false,
    dim: 1024,
    max_context_window: 32768,
    size_bytes: 0,
    source: "hf:unknown",
    is_default: false,
    comparator: {
      embedder_slug: "harrier-oss-v1-0.6b-MLX-4bit",
      entitled: true,
      exists: false,
      path: "comparator-harrier-600m-v1.json",
    },
  },
]

/** List all embedders (gui_embedders from triq.yaml). */
export async function listEmbedders(): Promise<GuiEmbedder[]> {
  try {
    return (await appInvoke("list_embedders", {})) as GuiEmbedder[]
  } catch (error) {
    // Without Tauri there is no serve loop to ask (the dev proxy needs a `triq`
    // binary), so fall back to the static catalog. Inside Tauri the bridge error
    // is real: surface it instead of faking an empty model list.
    if (!isTauri()) return BROWSER_EMBEDDERS
    throw error
  }
}

/** Download an embedder. */
export async function downloadEmbedder(slug: string): Promise<void> {
  return await appInvoke("download_embedder", { slug })
}

/** Remove a downloaded embedder. */
export async function removeEmbedder(slug: string): Promise<void> {
  return await appInvoke("remove_embedder", { slug })
}

/** Set the embedding model (switch engine to this model). */
export async function setEmbeddingModel(slug: string): Promise<void> {
  return await appInvoke("set_embedding_model", { slug })
}

/** Make an embedder the default (persists preference, no engine swap). */
export async function makeDefaultEmbedder(slug: string): Promise<void> {
  return await appInvoke("make_default_embedder", { slug })
}

/** Poll the download status for a model slug. */
export async function getDownloadStatus(slug: string): Promise<DownloadStatus> {
  return await appInvoke("download_status", { slug })
}

/** Check if download is complete for a model slug. */
export async function isDownloadComplete(slug: string): Promise<boolean> {
  return await appInvoke("download_complete", { slug })
}

/** Get quick model info. */
export async function getScoutModel(): Promise<ScoutModel | null> {
  try {
    return (await appInvoke("get_scout_model", {})) as ScoutModel | null
  } catch (error) {
    if (!isTauri()) return BROWSER_SCOUT
    throw error
  }
}

/** Load a comparator artifact from a path on disk. */
export async function loadComparatorArtifact(
  sourcePath: string,
  slug?: string
): Promise<void> {
  return await appInvoke("load_comparator_artifact", {
    sourcePath,
    embedderSlug: slug,
  })
}

/** Remove the comparator artifact for an embedder slug. */
export async function removeComparatorArtifact(slug: string): Promise<void> {
  return await appInvoke("remove_comparator_artifact", { slug })
}

/** Listen for model-download-progress events. */
export async function listenModelDownloadProgress(
  callback: (event: ModelDownloadProgress) => void
): Promise<UnlistenFn> {
  return await listenEvent<ModelDownloadProgress>(
    "model-download-progress",
    callback
  )
}

/** Listen for model-download-complete events. */
export async function listenModelDownloadComplete(
  callback: (event: ModelDownloadComplete) => void
): Promise<UnlistenFn> {
  return await listenEvent<ModelDownloadComplete>(
    "model-download-complete",
    callback
  )
}
