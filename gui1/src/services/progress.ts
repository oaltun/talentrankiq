import type { UnlistenFn } from "@tauri-apps/api/event"
import { listenEvent } from "../lib/events"
import type { EmbeddingProgress, RankProgress } from "../types"

export async function listenConversionProgress(
  callback: (event: EmbeddingProgress) => void
): Promise<UnlistenFn> {
  return await listenEvent<EmbeddingProgress>(
    "job-conversion-progress",
    callback
  )
}

export async function listenScoutProgress(
  callback: (event: EmbeddingProgress) => void
): Promise<UnlistenFn> {
  return await listenEvent<EmbeddingProgress>("job-quick-progress", callback)
}

export async function listenHeavyProgress(
  callback: (event: EmbeddingProgress) => void
): Promise<UnlistenFn> {
  return await listenEvent<EmbeddingProgress>("job-deep-progress", callback)
}

export async function listenRankProgress(
  callback: (event: RankProgress) => void
): Promise<UnlistenFn> {
  return await listenEvent<RankProgress>("rank-progress", callback)
}
