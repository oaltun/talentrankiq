import { useMutation } from '@tanstack/react-query'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type { ResultsCsvSummary } from '@/lib/tauri-bindings'

/** Write the ranked results (all ranked candidates, one column per question)
 * to `outPath` as CSV. Pure data export — no CV files, unlike the job bundle. */
export function useExportResultsCsv() {
  return useMutation({
    mutationFn: async (input: {
      jobId: string
      outPath: string
    }): Promise<ResultsCsvSummary> =>
      unwrapResult(await commands.exportResultsCsv(input.jobId, input.outPath)),
  })
}
