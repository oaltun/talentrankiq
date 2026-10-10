import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { useQueryClient } from '@tanstack/react-query'

import { candidatesQueryKeys } from '@/services/candidates'
import { jobsQueryKeys } from '@/services/jobs'

/**
 * Keep the job-ad card in step with the file on disk.
 *
 * The ad is a plain text file the user edits in their *own* editor, so the only
 * thing that can tell the gui its content moved is the backend's
 * `job-ad-reloaded` event. Called once, by the Ad file card, so the single card
 * reading `useJobAdFile` re-fetches from a single listener.
 *
 * The candidates query is invalidated too, and not just for tidiness: the backend
 * re-projects the job before it emits, so the sheet's totals have changed by the
 * time this fires.
 */
export function useJobAdReloaded(jobId: string) {
  const qc = useQueryClient()

  useEffect(() => {
    let disposed = false
    let unlisten: (() => void) | undefined
    listen<{ job_id: string }>('job-ad-reloaded', e => {
      if (e.payload?.job_id === jobId) {
        void qc.invalidateQueries({ queryKey: jobsQueryKeys.adFile(jobId) })
        void qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
      }
    }).then(fn => {
      if (disposed) fn()
      else unlisten = fn
    })
    return () => {
      disposed = true
      unlisten?.()
    }
  }, [jobId, qc])
}
