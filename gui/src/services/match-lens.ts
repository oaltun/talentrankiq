import { useQuery, useQueryClient } from '@tanstack/react-query'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type { ChunkMatchResult, ChunkMatchTarget } from '@/lib/tauri-bindings'
import { runtimeQueryKeys } from './runtime'

export const matchLensQueryKeys = {
  all: ['match-lens'] as const,
  chunk: (candidateId: string, target: ChunkMatchTarget, chunkSize: number) =>
    [
      ...matchLensQueryKeys.all,
      'chunks',
      candidateId,
      target,
      chunkSize,
    ] as const,
}

/**
 * Match Lens chunk-vs-target similarities. Heavy on-demand model work, so the
 * query is gated on the caller (`engineReady && hasText`); changing n never
 * refetches — n is applied client-side from the full chunk list.
 */
export function useChunkMatch(
  candidateId: string | null,
  target: ChunkMatchTarget,
  chunkSize: number,
  enabled: boolean
) {
  return useQuery({
    queryKey: matchLensQueryKeys.chunk(
      candidateId ?? '__none__',
      target,
      chunkSize
    ),
    queryFn: async (): Promise<ChunkMatchResult> =>
      unwrapResult(
        await commands.chunkMatch(candidateId as string, target, chunkSize)
      ),
    enabled: enabled && !!candidateId,
    // Chunk embeds can take a while; don't let a retry loop hammer the engine.
    retry: 1,
  })
}

/** Invalidate the engine status (used by the lens' inline retry button). */
export function useRetryRuntimeStatus() {
  const qc = useQueryClient()
  return () => qc.invalidateQueries({ queryKey: runtimeQueryKeys.status() })
}
