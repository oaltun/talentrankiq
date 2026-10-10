import { useMutation, useQueryClient } from '@tanstack/react-query'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type {
  BundleExportOptions,
  BundleExportSummary,
  Job,
} from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'

export function useExportJobBundle() {
  return useMutation({
    mutationFn: async (input: {
      jobId: string
      options: BundleExportOptions
      outPath: string
    }): Promise<BundleExportSummary> =>
      unwrapResult(
        await commands.exportJobBundle(
          input.jobId,
          input.options,
          input.outPath
        )
      ),
  })
}

/** Import a `.triquijob` bundle as a NEW job (fresh ids, staged CV files). */
export function useImportJobBundle() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (path: string): Promise<Job> =>
      unwrapResult(await commands.importJobBundle(path)),
    onSuccess: () => qc.invalidateQueries({ queryKey: jobsQueryKeys.all }),
  })
}
