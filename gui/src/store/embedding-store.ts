import { create } from 'zustand'

/** Live embedding progress for one job (driven by `job-embedding-progress` events). */
export interface EmbeddingProgress {
  done: number
  total: number
  failed: number
  active: boolean
}

interface EmbeddingState {
  progress: Record<string, EmbeddingProgress>
  setProgress: (jobId: string, p: EmbeddingProgress) => void
  clear: (jobId: string) => void
}

export const useEmbeddingStore = create<EmbeddingState>(set => ({
  progress: {},
  setProgress: (jobId, p) =>
    set(state => ({ progress: { ...state.progress, [jobId]: p } })),
  clear: jobId =>
    set(state => ({
      progress: Object.fromEntries(
        Object.entries(state.progress).filter(([key]) => key !== jobId)
      ),
    })),
}))
