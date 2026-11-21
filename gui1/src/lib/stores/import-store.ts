// Store for import progress per job.

import { create } from "zustand"
import type { ImportProgress } from "../../types"

interface ImportState {
  progressByJob: Record<string, ImportProgress>
  setProgress: (jobId: string, progress: ImportProgress) => void
  getProgress: (jobId: string) => ImportProgress | undefined
  clearProgress: (jobId: string) => void
}

export const useImportStore = create<ImportState>((set, get) => ({
  progressByJob: {},

  setProgress: (jobId: string, progress: ImportProgress) => {
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
