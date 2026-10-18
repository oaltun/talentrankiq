import { useCallback } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type { TriqiItem } from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'
import { candidatesQueryKeys } from './candidates'
import {
  useFeedbackStore,
  selectLikes,
  selectOrder,
} from '@/store/feedback-store'

export const feedbackQueryKeys = {
  all: ['feedback'] as const,
  likedIds: (jobId: string) =>
    [...feedbackQueryKeys.all, 'likedIds', jobId] as const,
}

/** All liked candidate ids for a job — seeds the staging store. */
export function useLikedCandidateIds(jobId: string) {
  return useQuery({
    queryKey: feedbackQueryKeys.likedIds(jobId),
    queryFn: async (): Promise<string[]> =>
      unwrapResult(await commands.listLikedCandidateIds(jobId)),
  })
}

/** Apply staged feedback (reordered triqui + desired liked set) and re-rank. */
export function useApplyFeedback(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: {
      orderedCandidateIds: string[]
      likedCandidateIds: string[]
    }) =>
      unwrapResult(
        await commands.applyFeedback(
          jobId,
          input.orderedCandidateIds,
          input.likedCandidateIds
        )
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
      qc.invalidateQueries({ queryKey: feedbackQueryKeys.all })
    },
  })
}

/** Zero the learned feedback vectors + clear liked flags, then re-rank. */
export function useResetFeedback(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => unwrapResult(await commands.resetFeedback(jobId)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
      qc.invalidateQueries({ queryKey: feedbackQueryKeys.all })
    },
  })
}

/**
 * Like/unlike a candidate AND immediately apply it as feedback (re-rank). Shared by
 * the grid and the detail overlay so liking from either place re-ranks right away.
 */
export function useLikeCandidate(jobId: string, triqui: TriqiItem[]) {
  const applyFeedback = useApplyFeedback(jobId)
  const toggleLike = useFeedbackStore(s => s.toggleLike)
  const clear = useFeedbackStore(s => s.clear)
  const stageOrder = useFeedbackStore(selectOrder(jobId))
  const likes = useFeedbackStore(selectLikes(jobId))

  return useCallback(
    (id: string) => {
      toggleLike(jobId, id) // instant star feedback
      const cur = Object.keys(likes).filter(x => likes[x])
      const nextLiked = cur.includes(id)
        ? cur.filter(x => x !== id)
        : [...cur, id]
      const order = stageOrder ?? triqui.map(s => s.id)
      applyFeedback.mutate(
        { orderedCandidateIds: order, likedCandidateIds: nextLiked },
        { onSuccess: () => clear(jobId) }
      )
    },
    [jobId, triqui, toggleLike, clear, stageOrder, likes, applyFeedback]
  )
}
