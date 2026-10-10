import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type {
  CandidateDetail,
  CandidateNeighbors,
  CandidatePage,
} from '@/lib/tauri-bindings'
import { jobsQueryKeys } from './jobs'

export const candidatesQueryKeys = {
  all: ['candidates'] as const,
  detail: (id: string) => [...candidatesQueryKeys.all, id] as const,
  page: (
    jobId: string,
    offset: number,
    limit: number,
    query: string,
    sortBy: string | null,
    sortDir: boolean,
    version?: unknown
  ) =>
    [
      ...candidatesQueryKeys.all,
      'page',
      jobId,
      offset,
      limit,
      query,
      sortBy,
      sortDir,
      version ?? 'v0',
    ] as const,
  neighbors: (
    jobId: string,
    id: string,
    query: string,
    sortBy: string | null,
    sortDir: boolean
  ) =>
    [
      ...candidatesQueryKeys.all,
      'neighbors',
      jobId,
      id,
      query,
      sortBy,
      sortDir,
    ] as const,
}

export function useCandidate(candidateId: string | null | undefined) {
  return useQuery({
    queryKey: candidatesQueryKeys.detail(candidateId ?? '__none__'),
    queryFn: async (): Promise<CandidateDetail> => {
      if (!candidateId) throw new Error('candidateId is required')
      return unwrapResult(await commands.getCandidate(candidateId))
    },
    enabled: !!candidateId,
  })
}

/** A page of a job's candidates, optionally filtered + sorted by a column
 * (`sortBy` = question id; null / '__total' = total points; '__name' / '__rank' =
 * file name / stored rank). `sortDir` = true → ascending. `version` is a
 * monotonic token (e.g. `rerank_dirty` or `ranked_version`) that forces a
 * refetch when the job's ranking changes. */
export function useCandidatesPage(
  jobId: string,
  offset: number,
  limit: number,
  query: string,
  sortBy: string | null,
  sortDir: boolean,
  version?: unknown
) {
  const normalized = query.trim().toLowerCase()
  return useQuery({
    queryKey: candidatesQueryKeys.page(
      jobId,
      offset,
      limit,
      normalized,
      sortBy,
      sortDir,
      version
    ),
    queryFn: async (): Promise<CandidatePage> =>
      unwrapResult(
        await commands.listCandidates(
          jobId,
          offset,
          limit,
          normalized || null,
          sortBy,
          sortDir
        )
      ),
    placeholderData: previous => previous,
  })
}

/** Prev/next candidate ids in the job's browser ordering (same sort as the page) for
 * overlay nav. */
export function useCandidateNeighbors(
  jobId: string,
  candidateId: string | null | undefined,
  query: string,
  sortBy: string | null,
  sortDir: boolean
) {
  const normalized = query.trim().toLowerCase()
  return useQuery({
    queryKey: candidatesQueryKeys.neighbors(
      jobId,
      candidateId ?? '__none__',
      normalized,
      sortBy,
      sortDir
    ),
    queryFn: async (): Promise<CandidateNeighbors> =>
      unwrapResult(
        await commands.candidateNeighbors(
          jobId,
          candidateId as string,
          normalized || null,
          sortBy,
          sortDir
        )
      ),
    enabled: !!candidateId,
  })
}

/** Delete every candidate for a job (keeps criteria) so the HR can re-import. */
export function useEmptyCandidates(jobId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => unwrapResult(await commands.emptyCandidates(jobId)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: jobsQueryKeys.detail(jobId) })
      qc.invalidateQueries({ queryKey: jobsQueryKeys.lists() })
      qc.invalidateQueries({ queryKey: candidatesQueryKeys.all })
    },
  })
}
