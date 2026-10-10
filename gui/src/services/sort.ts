import { useMutation, useQueryClient } from '@tanstack/react-query'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'
import { candidatesQueryKeys } from './candidates'

export function useSortJob(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => unwrapResult(await commands.sortJob(jobId)),
    onSuccess: () => {
      // Refresh the triqui and any open candidate overlay (scores/breakdown).
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
    },
  })
}
