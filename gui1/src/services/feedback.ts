// Frontend service for feedback (like/star) operations.

import { appInvoke } from "../lib/invoke"

/** Apply feedback (like/star) to candidates. */
export async function applyFeedback(
  jobId: string,
  orderedCandidateIds: string[],
  likedCandidateIds: string[]
): Promise<void> {
  return await appInvoke("apply_feedback", {
    jobId,
    orderedCandidateIds,
    likedCandidateIds,
  })
}

/** Reset all feedback for a job. */
export async function resetFeedback(jobId: string): Promise<void> {
  return await appInvoke("reset_feedback", { jobId })
}

/** List liked candidate IDs for a job. */
export async function listLikedCandidateIds(jobId: string): Promise<string[]> {
  return await appInvoke("list_liked_candidate_ids", { jobId })
}
