import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { listen } from '@tauri-apps/api/event'
import { useState, useEffect, useCallback, useRef } from 'react'
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

/**
 * At startup, checks whether the default embedding model's EFFECTIVE arm
 * (what `backend: auto` resolves to on this machine) is on disk. When it is
 * missing, the hook holds in `consent` — nothing downloads and the engine
 * does not start until the user agrees. Consent starts the shared
 * [`downloadEmbedder`] and reports progress for `ModelDownloadModal`; the
 * engine starts only once the model is usable (or on decline, with the
 * engine left unloaded — a later rank attempt surfaces engine-not-ready
 * instead of silently fetching gigabytes).
 */
export type StartupModelDownloadStatus =
  | 'checking'
  | 'consent'
  | 'downloading'
  | 'ready'
  | 'declined'
  | 'error'

export interface StartupConsent {
  model: GuiEmbedder
  variants: EmbedderVariants
}

export function useStartupModelDownload() {
  const qc = useQueryClient()
  const [status, setStatus] = useState<StartupModelDownloadStatus>('checking')
  const [model, setModel] = useState<GuiEmbedder | null>(null)
  const [consent, setConsent] = useState<StartupConsent | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Slug the user already consented to: a retry after a failed download
  // goes straight back to downloading instead of re-asking.
  const consentedSlug = useRef<string | null>(null)

  const startEngine = useCallback(() => {
    commands.startRuntime().catch(() => undefined)
  }, [])

  const run = useCallback(async () => {
    setStatus('checking')
    setError(null)
    setConsent(null)
    try {
      const embedders = unwrapResult(await commands.listEmbedders())
      const active = embedders.find(e => e.is_default)
      if (!active) {
        setStatus('ready')
        startEngine()
        return
      }
      // Consent keys on the EFFECTIVE arm: the onnx graph may be cached
      // while the resolved arm (e.g. MLX on macOS) is not. Ask the registry
      // which arm would serve and whether its weights exist; a failed
      // resolve falls back to trusting the downloaded flag.
      const variants = await fetchModelVariants(active.slug)
      if (!variants ? active.downloaded : variants.present) {
        setStatus('ready')
        startEngine()
        return
      }
      setModel(active)
      if (consentedSlug.current === active.slug) {
        // Already consented — resume the download (the retry path).
        setStatus('downloading')
        await downloadEmbedder(active.slug)
        qc.invalidateQueries({ queryKey: embeddersQueryKey })
        setStatus('ready')
        startEngine()
        return
      }
      setConsent({
        model: active,
        variants: variants ?? {
          model: active.slug,
          runtime: 'ort',
          present: false,
          onnx_present: active.downloaded,
          mlx_present: null,
        },
      })
      setStatus('consent')
    } catch (e) {
      setError(errorMessage(e))
      setStatus('error')
    }
  }, [qc, startEngine])

  const accept = useCallback(async () => {
    if (!consent) return
    consentedSlug.current = consent.model.slug
    setStatus('downloading')
    setError(null)
    try {
      await downloadEmbedder(consent.model.slug)
      qc.invalidateQueries({ queryKey: embeddersQueryKey })
      setStatus('ready')
      startEngine()
    } catch (e) {
      setError(errorMessage(e))
      setStatus('error')
    }
  }, [consent, qc, startEngine])

  const decline = useCallback(() => {
    setConsent(null)
    setStatus('declined')
  }, [])

  useEffect(() => {
    // Deferred off the synchronous effect body: run() flips the status state,
    // and a sync setState here would cascade renders (react-hooks/set-state-in-effect).
    const id = setTimeout(() => void run(), 0)
    return () => clearTimeout(id)
  }, [run])

  return { status, model, consent, error, retry: run, accept, decline }
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
      setProgress(p => ({
        ...p,
        [repo_id]: { repoId: repo_id, filename, done, total },
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
        if (k !== slug) next[k] = v
      }
      return next
    })
  const get = (slug: string): DownloadProgressEntry | undefined =>
    progress[slug]

  return { get, clear }
}
