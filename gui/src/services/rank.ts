import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { listen } from '@tauri-apps/api/event'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type { RankState } from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'
import { candidatesQueryKeys } from './candidates'

/** How often to poll rank state while a cascade is running. */
const POLL_MS = 1000

function rankStateKey(jobId: string) {
  return ['rank-state', jobId] as const
}

/**
 * Live rank state for a job (`undefined` until the first poll resolves).
 *
 * Polls only while the backend reports a run in flight, so an idle job costs one
 * query, not a timer. Progress is *reported* state — stage, wave, embedded
 * counts, converged — not a client-side guess; the backend owns it
 * (progressive-ranking design §10).
 */
export function useRankState(jobId: string) {
  return useQuery({
    queryKey: rankStateKey(jobId),
    queryFn: async () => unwrapResult(await commands.getRankState(jobId)),
    refetchInterval: query =>
      query.state.data?.running === true ? POLL_MS : false,
  })
}

/**
 * Drive the progressive-ranking cascade for a job.
 *
 * `startRank` returns immediately and the ranking converges in the background,
 * so the button must not block on it: the sheet is readable from stage 0 on and
 * improves underneath. The sheet is refetched on every `rank-progress` event, so
 * rows appear and reorder as waves land rather than all at the end.
 */
export function useRankJob(jobId: string) {
  const qc = useQueryClient()
  const rankState = useRankState(jobId)

  useEffect(() => {
    if (jobId === '__none__') return
    let disposed = false
    // Coalesce bursts: during a wave the backend reports embed progress per
    // candidate (backend-throttled to ~10 Hz) — refetching the whole sheet
    // per event starves the webview (clicks and native dialogs go dead).
    // One trailing refetch per quiet gap keeps the sheet live at a sane
    // rate whatever the event rate.
    let timer: ReturnType<typeof setTimeout> | null = null
    const unlisten = listen('rank-progress', event => {
      if (disposed) return
      const payload = event.payload as { job_id?: string } | null
      if (payload?.job_id && payload.job_id !== jobId) return
      if (timer) return
      timer = setTimeout(() => {
        timer = null
        qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
        qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
        void qc.invalidateQueries({ queryKey: rankStateKey(jobId) })
      }, 250)
    })
    return () => {
      disposed = true
      if (timer) clearTimeout(timer)
      void unlisten.then(fn => fn())
    }
  }, [jobId, qc])

  return {
    ...useMutation({
      mutationFn: async () => unwrapResult(await commands.startRank(jobId)),
      onSuccess: () => {
        void qc.invalidateQueries({ queryKey: rankStateKey(jobId) })
      },
    }),
    state: rankState.data as RankState | undefined,
  }
}

/** Stop a running cascade. Safe to call when nothing is running. */
export function useCancelRank(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => unwrapResult(await commands.cancelRank(jobId)),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: rankStateKey(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
    },
  })
}
