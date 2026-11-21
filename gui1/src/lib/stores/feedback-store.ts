// Store for feedback (like/star) state per job.

import { create } from "zustand"

interface FeedbackState {
  likedIdsByJob: Record<string, string[]>
  setLikedIds: (jobId: string, ids: string[]) => void
  getLikedIds: (jobId: string) => string[]
  toggleLiked: (jobId: string, candidateId: string) => void
  // For React component to get likedIds for a specific job
  useLikedIds: (jobId: string) => string[]
}

export const useFeedbackStore = create<FeedbackState>((set, get) => ({
  likedIdsByJob: {},

  setLikedIds: (jobId: string, ids: string[]) => {
    set((state) => ({
      likedIdsByJob: { ...state.likedIdsByJob, [jobId]: ids },
    }))
  },

  getLikedIds: (jobId: string) => {
    return get().likedIdsByJob[jobId] ?? []
  },

  toggleLiked: (jobId: string, candidateId: string) => {
    set((state) => {
      const current = state.likedIdsByJob[jobId] ?? []
      const newIds = current.includes(candidateId)
        ? current.filter((id) => id !== candidateId)
        : [...current, candidateId]
      return { likedIdsByJob: { ...state.likedIdsByJob, [jobId]: newIds } }
    })
  },

  useLikedIds: (jobId: string) => {
    return get().likedIdsByJob[jobId] ?? []
  },
}))
