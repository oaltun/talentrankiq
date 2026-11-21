// Store for embedding progress per job.

import { create } from "zustand"
import type { EmbeddingProgress } from "../../types"

interface EmbeddingState {
  progressByJob: Record<string, EmbeddingProgress>
  setProgress: (jobId: string, progress: EmbeddingProgress) => void
  getProgress: (jobId: string) => EmbeddingProgress | undefined
  clearProgress: (jobId: string) => void
}

export const useEmbeddingStore = create<EmbeddingState>((set, get) => ({
  progressByJob: {},

  setProgress: (jobId: string, progress: EmbeddingProgress) => {
    set((state) => ({
      progressByJob: { ...state.progressByJob, [jobId]: progress },
    }))
  },

  getProgress: (jobId: string) => {
    return get().progressByJob[jobId]
  },

  clearProgress: (jobId: string) => {
    set((state) => {
      const rest = { ...state.progressByJob }
      delete rest[jobId]
      return { progressByJob: rest }
    })
  },
}))
