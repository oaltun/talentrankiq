import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { useQueryClient } from '@tanstack/react-query'

import {
  useEmbeddingStore,
  type EmbeddingProgress,
} from '@/store/embedding-store'
import { jobsQueryKeys } from '@/services/jobs'
import { candidatesQueryKeys } from '@/services/candidates'

/** Shape emitted by the embedding backfill/sort (plain string event — not specta-typed). */
interface EmbeddingProgressEvent {
  job_id: string
  done: number
  total: number
  failed: number
  active: boolean
}

/**
 * Subscribe once (mount in the shell) to `job-embedding-progress` events and mirror
 * them into the embedding store. On completion it refreshes the affected queries.
 */
export function useEmbeddingProgressListener() {
  const qc = useQueryClient()
  const setProgress = useEmbeddingStore(s => s.setProgress)
  const clear = useEmbeddingStore(s => s.clear)

  useEffect(() => {
    let mounted = true
    let unlisten: (() => void) | null = null

    listen<EmbeddingProgressEvent>('job-embedding-progress', event => {
      const { job_id, done, total, failed, active } = event.payload
      const p: EmbeddingProgress = { done, total, failed, active }
      setProgress(job_id, p)

      if (!active) {
        // Embedding finished — refresh persisted data (candidates too: ranks may have
        // changed), then clear the live badge.
        qc.invalidateQueries({ queryKey: jobsQueryKeys.all })
        qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
        window.setTimeout(() => clear(job_id), 1500)
      }
    })
      .then(unlistenFn => {
        if (mounted) unlisten = unlistenFn
        else unlistenFn()
      })
      .catch(error => {
        // Non-fatal: embedding progress just won't show live.
        console.error('Failed to listen for embedding progress', error)
      })

    return () => {
      mounted = false
      if (unlisten) unlisten()
    }
  }, [qc, setProgress, clear])
}
