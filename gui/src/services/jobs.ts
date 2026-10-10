import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type {
  Job,
  JobAdFile,
  JobDetail,
  JobSummary,
  JobUiState,
} from '@/lib/tauri-bindings'
import { candidatesQueryKeys } from './candidates'

export const jobsQueryKeys = {
  all: ['jobs'] as const,
  lists: () => [...jobsQueryKeys.all, 'list'] as const,
  list: () => [...jobsQueryKeys.lists()] as const,
  details: () => [...jobsQueryKeys.all, 'detail'] as const,
  detail: (id: string) => [...jobsQueryKeys.details(), id] as const,
  uiState: (id: string) => [...jobsQueryKeys.detail(id), 'ui-state'] as const,
  adFile: (id: string) => [...jobsQueryKeys.detail(id), 'ad-file'] as const,
}

export function useJobs() {
  return useQuery({
    queryKey: jobsQueryKeys.list(),
    queryFn: async (): Promise<JobSummary[]> =>
      unwrapResult(await commands.listJobSummaries()),
  })
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: jobsQueryKeys.detail(jobId ?? '__none__'),
    queryFn: async (): Promise<JobDetail> => {
      if (!jobId) throw new Error('jobId is required')
      return unwrapResult(await commands.getJobDetail(jobId))
    },
    enabled: !!jobId,
  })
}

/**
 * The job-ad **file**: its path and contents. Asking for it also *resolves* the
 * default location (`<cv-root>/<job-slug>/job-ad.txt`), creating it empty, and
 * arms the watcher that re-reads it on disk changes — so this is the one call
 * that makes the pane work, and the backend owns the path.
 */
export function useJobAdFile(jobId: string | undefined) {
  return useQuery({
    queryKey: jobsQueryKeys.adFile(jobId ?? '__none__'),
    queryFn: async (): Promise<JobAdFile> => {
      if (!jobId) throw new Error('jobId is required')
      return unwrapResult(await commands.getJobAd(jobId))
    },
    enabled: !!jobId,
  })
}

/** "Change…": point the job at a different ad file. */
export function useSetJobAdPath(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (path: string) =>
      unwrapResult(await commands.setJobAdPath(jobId, path)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.adFile(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
    },
  })
}

/**
 * Backend-derived enablement for the Sort / Re-rank buttons: the serve owns
 * `sort_allowed`, dirtiness, and the reason code (`no-cvs`, `no-jobad`,
 * `engine-not-ready`, `stale`, `clean`); the gui only i18n-maps the reason.
 * Polled so engine readiness flips surface without a dedicated event bus.
 */
export function useJobUiState(jobId: string | undefined) {
  return useQuery({
    queryKey: jobsQueryKeys.uiState(jobId ?? '__none__'),
    queryFn: async (): Promise<JobUiState> => {
      if (!jobId) throw new Error('jobId is required')
      return unwrapResult(await commands.jobUiState(jobId))
    },
    enabled: !!jobId,
    refetchInterval: 2000,
  })
}

export function useCreateJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (): Promise<Job> =>
      unwrapResult(await commands.createJob()),
    onSuccess: () => qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() }),
  })
}

export interface JobUpdateInput {
  title: string
  jobAdText: string | null
  triquiSize: number
}

export function useUpdateJob(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: JobUpdateInput) =>
      unwrapResult(
        await commands.updateJob(
          jobId,
          input.title,
          input.jobAdText,
          input.triquiSize
        )
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
      // The backend re-projected on the way out (scoring v3), so the sheet is
      // showing the old totals until the candidates query refetches.
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
    },
  })
}

/** Update the rubric weights (job-ad + the two feedback rows). Ranking inputs, and
 * like a criterion edit they are re-projected server-side rather than left stale. */
export function useUpdateJobWeights(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: {
      adWeight: number
      fbJobWeight: number
      fbProfileWeight: number
    }) =>
      unwrapResult(
        await commands.updateJobWeights(
          jobId,
          input.adWeight,
          input.fbJobWeight,
          input.fbProfileWeight
        )
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
    },
  })
}

/** Set (or clear with '') the job's watched CV directory; restarts the watcher.
 * New CVs dropped there are imported automatically. */
export function useSetCvDir(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (directory: string) =>
      unwrapResult(await commands.setJobCvDir(jobId, directory)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
    },
  })
}

export function useArchiveJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: { jobId: string; archived: boolean }) =>
      unwrapResult(await commands.archiveJob(input.jobId, input.archived)),
    onSuccess: () => qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() }),
  })
}

export function useDeleteJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (jobId: string) =>
      unwrapResult(await commands.deleteJob(jobId)),
    onSuccess: () => qc.invalidateQueries({ queryKey: jobsQueryKeys.all }),
  })
}
