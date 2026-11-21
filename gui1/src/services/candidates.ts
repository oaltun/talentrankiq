// Frontend service for candidate management.

import { appInvoke } from "../lib/invoke"
import type {
  CandidatePage,
  CandidateSummary,
  CandidateDetail,
  CandidateNeighbors,
} from "../types"

export type {
  CandidatePage,
  CandidateSummary,
  CandidateDetail,
  CandidateNeighbors,
}

/** List candidates with pagination, search, and sorting. */
export async function listCandidates(
  jobId: string,
  offset: number,
  limit: number,
  query?: string,
  sortBy?: string,
  sortDir?: boolean
): Promise<CandidatePage> {
  return await appInvoke("list_candidates", {
    jobId,
    offset,
    limit,
    query,
    sortBy,
    sortDir,
  })
}

/** Get a single candidate detail (for CV Inspector). */
export async function getCandidate(
  candidateId: string
): Promise<CandidateDetail> {
  return await appInvoke("get_candidate", { candidateId })
}

/** Get prev/next candidate IDs for overlay navigation. */
export async function candidateNeighbors(
  jobId: string,
  candidateId: string,
  query?: string,
  sortBy?: string,
  sortDir?: boolean
): Promise<CandidateNeighbors> {
  return await appInvoke("candidate_neighbors", {
    jobId,
    candidateId,
    query,
    sortBy,
    sortDir,
  })
}

/** Apply feedback (like/star) to candidates. */
export async function applyFeedback(
  jobId: string,
  orderedCandidateIds: string[],
  likedCandidateIds: string[]
): Promise<void> {
  await appInvoke("apply_feedback", {
    jobId,
    orderedCandidateIds,
    likedCandidateIds,
  })
}

/** Reset all feedback for a job. */
export async function resetFeedback(jobId: string): Promise<void> {
  await appInvoke("reset_feedback", { jobId })
}

/** List liked candidate IDs for a job. */
export async function listLikedCandidateIds(jobId: string): Promise<string[]> {
  return await appInvoke("list_liked_candidate_ids", { jobId })
}
