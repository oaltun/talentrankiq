// Frontend service for Match Lens operations.

import { appInvoke } from "../lib/invoke"
import type {
  ChunkMatchResult,
  ChunkMatchTarget,
  CandidateDetail,
} from "../types"

export type { ChunkMatchResult, ChunkMatchTarget, CandidateDetail }

/** Get Match Lens chunks for a candidate. */
export async function chunkMatch(
  candidateId: string,
  target: ChunkMatchTarget,
  chunkSize: number
): Promise<ChunkMatchResult> {
  return await appInvoke("chunk_match", {
    candidateId,
    target,
    chunkSize,
  })
}
