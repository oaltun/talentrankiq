import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect, useCallback } from 'react'
import { toast } from 'sonner'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { errorMessage } from '@/lib/errors'
import type {
  DownloadStatus,
  EmbedderVariants,
  GuiEmbedder,
  ModelKind,
} from '@/lib/tauri-bindings'
import { jobsQueryKeys } from '@/services/jobs'
import { candidatesQueryKeys } from '@/services/candidates'
import { pilotShim } from '@/lib/pilot-shim'

export const embeddersQueryKey = ['embedders'] as const

export type { ModelKind }
export type { GuiEmbedder }
export type { EmbedderVariants }

/**
 * The ONE embedder download used everywhere: the Models activity and the
 * startup modal both go through this (serve-side it is a single code path
 * that emits `model-download-progress` events).
 */
export async function downloadEmbedder(slug: string): Promise<void> {
  await unwrapResult(await commands.downloadEmbedder(slug))
}

/**
 * Which arm (`backend: auto`) would serve a slug and whether its weights are
 * on disk — engine-free registry resolve. `null` when the backend predates
 * the command or the slug is unknown; callers fall back to the downloaded
 * flag in that case.
 */
export async function fetchModelVariants(
  slug: string
): Promise<EmbedderVariants | null> {
  try {
    return unwrapResult(await commands.modelVariants(slug))
  } catch {
    return null
  }
}

/** Poll-friendly variants query for a card/badge. */
export function useModelVariants(slug: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['model-variants', slug],
    queryFn: async (): Promise<EmbedderVariants | null> =>
      fetchModelVariants(slug ?? ''),
    enabled: enabled && !!slug,
    staleTime: 1000 * 30,
  })
}

/** Payload for the `model-download-progress` string event (not specta-typed). */
export interface ModelDownloadProgress {
  repo_id: string
  filename: string
  done: number
  total: number | null
}

/** Hook that loads the curated ONNX embedder list (`gui_embedders`). */
export function useEmbedders(enabled = true) {
  return useQuery({
    queryKey: embeddersQueryKey,
    queryFn: async (): Promise<GuiEmbedder[]> =>
      unwrapResult(await commands.listEmbedders()),
    enabled,
    staleTime: 1000 * 30,
  })
}

/** Download an ONNX embedder via the shared [`downloadEmbedder`], invalidating the list after. */
export function useDownloadEmbedder() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: downloadEmbedder,
    onSuccess: () => {
      toast.success('Download complete')
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
    },
    onError: (e: unknown) => {
      toast.error('Download failed', {
        description: errorMessage(e),
      })
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
    },
  })
}

/** Remove a downloaded ONNX embedder from the cache. */
export function useRemoveEmbedder() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (slug: string) =>
      unwrapResult(await commands.removeEmbedder(slug)),
    onSuccess: () => {
      toast('Removed model')
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
    },
    onError: (e: unknown) => {
      toast.error('Remove failed', {
        description: errorMessage(e),
      })
    },
  })
}

/**
 * Switch the active embedding model (persists the choice, hot-swaps the engine,
 * marks every job's triqui stale — all CVs re-embed on their next rank).
 */
export function useSetEmbeddingModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (slug: string) =>
      unwrapResult(await commands.setEmbeddingModel(slug)),
    onSuccess: () => {
      toast.success('Embedding model switched')
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
      qc.invalidateQueries({ queryKey: ['preferences'] })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.all })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
    },
    onError: (e: unknown) => {
      toast.error('Could not switch embedding model', {
        description: errorMessage(e),
      })
    },
  })
}

/** Mark an embedder as the default (persisted to preferences, applied live). */
export function useSetDefaultEmbedder() {
  const qc = useQueryClient()
  const setEmbeddingModel = useSetEmbeddingModel()
  return useMutation({
    mutationFn: async (slug: string) =>
      unwrapResult(
        await commands.setDefaultModel('embedding' as ModelKind, slug)
      ),
    onSuccess: (_, slug) => {
      // Apply the new default to the running engine too.
      setEmbeddingModel.mutate(slug)
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
      qc.invalidateQueries({ queryKey: ['preferences'] })
    },
    onError: (e: unknown) => {
      toast.error('Could not set default model', {
        description: errorMessage(e),
      })
    },
  })
}

/** Listen for `model-download-progress` events. Mount in the models tab. */
interface DownloadProgressEntry {
  repoId: string
  filename: string
  done: number
  total: number | null
}
export function useModelDownloadProgress() {
  const [progress, setProgress] = useState<
    Record<string, DownloadProgressEntry>
  >({})

  const handleProgress = useCallback((payload: unknown) => {
    const { repo_id, filename, done, total } = payload as {
      repo_id: string
      filename: string
      done: number
      total: number | null
    }
    const key = repo_id
    setProgress(p => ({
      ...p,
      [key]: { repoId: repo_id, filename, done, total },
    }))
  }, [])

  useEffect(() => {
    let mounted = true

    const setupListener = async () => {
      try {
        await pilotShim.on('model-download-progress', handleProgress)
      } catch (e) {
        console.error(
          '[useModelDownloadProgress] Failed to listen via pilotShim:',
          e
        )
        // Fallback to native Tauri events (for production)
        try {
          const { listen } = await import('@tauri-apps/api/event')
          const unlisten = await listen('model-download-progress', event => {
            if (!mounted) return
            handleProgress(event.payload)
          })
          return () => unlisten()
        } catch (fallbackErr) {
          console.error(
            '[useModelDownloadProgress] Fallback listen also failed:',
            fallbackErr
          )
        }
      }
    }

    setupListener()

    return () => {
      mounted = false
      pilotShim.off('model-download-progress', handleProgress).catch(e => {
        console.error('[useModelDownloadProgress] Failed to off:', e)
      })
    }
  }, [handleProgress])

  const clear = (slug: string) =>
    setProgress(p => {
      const next: Record<string, DownloadProgressEntry> = {}
      for (const [k, v] of Object.entries(p)) {
        if (k !== slug) next[k] = v
      }
      return next
    })
  const get = (slug: string): DownloadProgressEntry[] =>
    Object.values(progress).filter(e => e.repoId === slug)

  return { get, clear }
}

/** Listen for `model-download-complete` events and show a toast with the save path. */
export function useModelDownloadCompleteToast() {
  useEffect(() => {
    const handleComplete = (payload: unknown) => {
      const { path } = payload as { path: string }
      toast.success('Model downloaded', {
        description: `Saved to ${path}`,
      })
    }

    let mounted = true

    const setupListener = async () => {
      try {
        await pilotShim.on('model-download-complete', handleComplete)
      } catch (e) {
        console.error(
          '[useModelDownloadCompleteToast] Failed to listen via pilotShim:',
          e
        )
        // Fallback to native Tauri events (for production)
        try {
          const { listen } = await import('@tauri-apps/api/event')
          const unlisten = await listen('model-download-complete', event => {
            if (!mounted) return
            handleComplete(event.payload)
          })
          return () => unlisten()
        } catch (fallbackErr) {
          console.error(
            '[useModelDownloadCompleteToast] Fallback listen also failed:',
            fallbackErr
          )
        }
      }
    }

    setupListener()

    return () => {
      mounted = false
      pilotShim.off('model-download-complete', handleComplete).catch(e => {
        console.error('[useModelDownloadCompleteToast] Failed to off:', e)
      })
    }
  }, [])
}

/** Poll the download status for a model slug every 1 second. */
export function useDownloadStatus(slug: string | undefined, enabled = true) {
  const [status, setStatus] = useState<DownloadStatus | null>(null)

  useEffect(() => {
    if (!enabled || !slug) return

    let mounted = true
    let intervalId: ReturnType<typeof setInterval> | null = null
    let seenProgress = false

    const fetchStatus = async () => {
      try {
        const result = await commands.downloadStatus(slug)
        if (!mounted) return
        const data = unwrapResult(result)

        // Check if we've seen actual progress (bytes downloaded)
        const hasProgress = data.done_bytes > 0 || data.total_bytes !== null

        if (hasProgress) {
          seenProgress = true
        }

        // If download is complete AND we've seen progress, clear status and stop
        if (!data.is_downloading && seenProgress) {
          setStatus(null)
          if (intervalId) clearInterval(intervalId)
          return
        }

        // If we haven't seen progress yet, keep polling (backend may not have created entry yet)
        // If we have seen progress and download is still active, update status
        if (data.is_downloading || !seenProgress) {
          setStatus(data)
        }
      } catch (e) {
        console.error('[useDownloadStatus] Failed to fetch status:', e)
        if (!mounted) return
        setStatus(null)
        if (intervalId) clearInterval(intervalId)
      }
    }

    const checkComplete = async () => {
      try {
        const result = await commands.downloadComplete(slug)
        if (!mounted) return
        const data = unwrapResult(result)
        if (data) {
          // Download is complete - clear status and stop polling
          setStatus(null)
          if (intervalId) clearInterval(intervalId)
        }
      } catch (e) {
        console.error('[useDownloadStatus] Failed to check completion:', e)
      }
    }

    // Initial fetch
    fetchStatus()

    // Poll status every 1 second
    intervalId = setInterval(fetchStatus, 1000)

    // Also check completion every 1 second (offset by 500ms)
    const completeIntervalId = setInterval(checkComplete, 1000)
    // Initial check after 500ms
    setTimeout(checkComplete, 500)

    return () => {
      mounted = false
      if (intervalId) clearInterval(intervalId)
      if (completeIntervalId) clearInterval(completeIntervalId)
    }
  }, [slug, enabled])

  return status
}
