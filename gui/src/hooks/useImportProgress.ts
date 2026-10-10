import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { useQueryClient } from '@tanstack/react-query'

import { useImportStore, type ImportProgress } from '@/store/import-store'
import { jobsQueryKeys } from '@/services/jobs'
import { candidatesQueryKeys } from '@/services/candidates'
import { toast } from 'sonner'

/** Shape emitted by `commands.importCvDirectory` (plain string event — not specta-typed). */
interface ImportProgressEvent {
  job_id: string
  done: number
  total: number
  failed: number
  active: boolean
  phase: string
}

/**
 * Subscribe once (mount in the shell) to `job-import-progress` events and mirror
 * them into the import store. On completion it refreshes the affected queries.
 */
export function useImportProgressListener() {
  const qc = useQueryClient()
  const setProgress = useImportStore(s => s.setProgress)
  const clear = useImportStore(s => s.clear)

  useEffect(() => {
    let mounted = true
    let unlisten: (() => void) | null = null

    listen<ImportProgressEvent>('job-import-progress', event => {
      const { job_id, done, total, failed, active, phase } = event.payload
      const p: ImportProgress = { done, total, failed, active, phase }
      setProgress(job_id, p)

      if (!active) {
        // Import finished — refresh persisted data (candidates too: an auto-sort may
        // have ranked them), then clear the live badge.
        qc.invalidateQueries({ queryKey: jobsQueryKeys.all })
        qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
        if (phase === 'engine-unavailable') {
          toast.warning(
            'AI engine was not ready, so CVs were imported but not embedded or ranked. ' +
              'Run Re-rank once the engine is up.'
          )
        }
        window.setTimeout(() => clear(job_id), 1500)
      }
    })
      .then(unlistenFn => {
        if (mounted) unlisten = unlistenFn
        else unlistenFn()
      })
      .catch(error => {
        // Non-fatal: import progress just won't show live.
        console.error('Failed to listen for import progress', error)
      })

    return () => {
      mounted = false
      if (unlisten) unlisten()
    }
  }, [qc, setProgress, clear])
}
