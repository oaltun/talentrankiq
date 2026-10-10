import { useMutation, useQueryClient } from '@tanstack/react-query'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'
import { candidatesQueryKeys } from './candidates'

/**
 * A criterion edit re-projects the job server-side, so it invalidates the **sheet**
 * as well as the job detail: the backend has already recomputed every candidate's
 * total by the time this returns, and the ranking on screen is stale until the
 * candidates query refetches. Before scoring v3 this was not needed — an edit left
 * the job marked stale and a Re-rank click was required — so the sheet was
 * correctly left alone.
 */
function invalidate(qc: ReturnType<typeof useQueryClient>, jobId: string) {
  qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
  qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
  qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
}

export interface CriterionInput {
  text: string
  /** Omit to use the configured default (`rubric_defaults.criterion_points`). */
  weight?: number
}

export function useAddCriterion(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: CriterionInput) =>
      unwrapResult(
        await commands.addCriterion(jobId, input.text, input.weight ?? null)
      ),
    onSuccess: () => invalidate(qc, jobId),
  })
}

export function useUpdateCriterion(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: { id: string; text: string; weight: number }) =>
      unwrapResult(
        await commands.updateCriterion(input.id, input.text, input.weight)
      ),
    onSuccess: () => invalidate(qc, jobId),
  })
}

export function useDeleteCriterion(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: string) =>
      unwrapResult(await commands.deleteCriterion(id)),
    onSuccess: () => invalidate(qc, jobId),
  })
}
