import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { listen } from '@tauri-apps/api/event'
import { commands } from '@/lib/tauri-bindings'

import { useRankJob } from './rank'

// The rank-progress wiring: every event refetches the sheet (job detail +
// candidates + rank state). During a wave the backend reports per-candidate
// embed progress, so bursts are the norm — an uncoalesced refetch per event
// starved the webview (clicks and native dialogs went dead until the embed
// ended). These tests pin the burst → one-refetch coalescing.

type ProgressHandler = (event: { payload: unknown }) => void
const progressHandlers = new Map<string, ProgressHandler>()

// The global setup mock resolves without capturing — capture handlers here.
vi.mocked(listen).mockImplementation(async (event, handler) => {
  progressHandlers.set(event as string, handler as unknown as ProgressHandler)
  return () => progressHandlers.delete(event as string)
})

// The global bindings mock has no getRankState — add it so useRankState's
// poll resolves while the run reports in flight.
const mockCommands = commands as unknown as Record<
  string,
  ReturnType<typeof vi.fn>
>

function emitRankProgress(jobId: string, done: number) {
  const handler = progressHandlers.get('rank-progress')
  if (!handler) throw new Error('useRankJob is not listening for rank-progress')
  handler({ payload: { job_id: jobId, stage: 'embed', embedded_done: done } })
}

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return { queryClient, wrapper }
}

beforeEach(() => {
  progressHandlers.clear()
  mockCommands.getRankState ??= vi.fn()
  mockCommands.getRankState.mockResolvedValue({
    status: 'ok',
    data: {
      running: true,
      ranking: false,
      stage: 'embed',
      wave: 1,
      embedded_total: 10,
      embedded_done: 1,
      judged: 0,
      candidates: 10,
      converged: false,
      reason: null,
    },
  })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useRankJob rank-progress handling', () => {
  it('coalesces an event burst into a single sheet refetch', async () => {
    const { queryClient, wrapper } = makeWrapper()
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    const { unmount } = renderHook(() => useRankJob('job-1'), { wrapper })
    await waitFor(() =>
      expect(progressHandlers.get('rank-progress')).toBeDefined()
    )
    vi.useFakeTimers()

    // A burst of per-candidate progress: nothing refetches until the quiet
    // gap elapses.
    for (let done = 1; done <= 10; done++) emitRankProgress('job-1', done)
    expect(invalidate).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(300)
    })
    // One refetch = detail + candidates + rank state.
    expect(invalidate).toHaveBeenCalledTimes(3)

    // The next burst after the gap refetches once more.
    emitRankProgress('job-1', 11)
    await act(async () => {
      vi.advanceTimersByTime(300)
    })
    expect(invalidate).toHaveBeenCalledTimes(6)

    unmount()
  })

  it('ignores events for other jobs', async () => {
    const { queryClient, wrapper } = makeWrapper()
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries')
    const { unmount } = renderHook(() => useRankJob('job-1'), { wrapper })
    await waitFor(() =>
      expect(progressHandlers.get('rank-progress')).toBeDefined()
    )
    vi.useFakeTimers()

    emitRankProgress('job-2', 1)
    emitRankProgress('job-2', 2)
    await act(async () => {
      vi.advanceTimersByTime(500)
    })
    expect(invalidate).not.toHaveBeenCalled()

    unmount()
  })
})
