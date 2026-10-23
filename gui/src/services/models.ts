import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { listen } from '@tauri-apps/api/event'
import { useState, useEffect } from 'react'
import { toast } from 'sonner'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { errorMessage } from '@/lib/errors'
import type {
  EmbedderVariants,
  GuiEmbedder,
  ModelKind,
} from '@/lib/tauri-bindings'
import { jobsQueryKeys } from '@/services/jobs'
import { candidatesQueryKeys } from '@/services/candidates'

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

  useEffect(() => {
    let unlisten: (() => void) | null = null
    listen<ModelDownloadProgress>('model-download-progress', event => {
      const { repo_id, filename, done, total } = event.payload
      // Key by repo_id + filename to track multiple files per model
      const key = `${repo_id}:${filename}`
      setProgress(p => ({
        ...p,
        [key]: { repoId: repo_id, filename, done, total },
      }))
    })
      .then(fn => {
        unlisten = fn
      })
      .catch(err =>
        console.error('Failed to listen for download progress', err)
      )

    return () => {
      unlisten?.()
    }
  }, [])

  const clear = (slug: string) =>
    setProgress(p => {
      const next: Record<string, DownloadProgressEntry> = {}
      for (const [k, v] of Object.entries(p)) {
        if (!k.startsWith(`${slug}:`)) next[k] = v
      }
      return next
    })
  const get = (slug: string): DownloadProgressEntry[] =>
    Object.values(progress).filter(e => e.repoId === slug)

  return { get, clear }
}

interface ModelDownloadCompleteEvent {
  slug: string
  path: string
}

/** Listen for `model-download-complete` events and show a toast with the save path. */
export function useModelDownloadCompleteToast() {
  useEffect(() => {
    let unlisten: (() => void) | null = null
    listen<ModelDownloadCompleteEvent>('model-download-complete', event => {
      const { path } = event.payload
      toast.success('Model downloaded', {
        description: `Saved to ${path}`,
      })
    })
      .then(fn => {
        unlisten = fn
      })
      .catch(err =>
        console.error('Failed to listen for download complete', err)
      )

    return () => {
      unlisten?.()
    }
  }, [])
}
