import { create } from 'zustand'

/** Live import progress for one job (driven by `job-import-progress` events). */
export interface ImportProgress {
  done: number
  total: number
  failed: number
  active: boolean
  /** "extracting" | "embedding" | "ranking" | "engine-unavailable" | "done" */
  phase: string
}

interface ImportState {
  progress: Record<string, ImportProgress>
  setProgress: (jobId: string, p: ImportProgress) => void
  clear: (jobId: string) => void
}

export const useImportStore = create<ImportState>(set => ({
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
