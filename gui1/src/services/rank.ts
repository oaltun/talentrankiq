// Frontend service for ranking operations.

import { appInvoke } from "../lib/invoke"
import type { RankState, JobUiState, SortReport } from "../types"

/** Get backend-owned enablement state for a job's Re-rank button. */
export async function getJobUiState(jobId: string): Promise<JobUiState> {
  return await appInvoke("job_ui_state", { jobId })
}

/** Get the current rank state for a job. */
export async function getRankState(jobId: string): Promise<RankState> {
  return await appInvoke("get_rank_state", { jobId })
}

/** Start a progressive-ranking cascade run for a job. */
export async function startRank(jobId: string): Promise<void> {
  return await appInvoke("start_rank", { jobId })
}

/** Start the new process_in_waves ranking pipeline for a job. */
export async function processInWaves(jobId: string): Promise<void> {
  return await appInvoke("process_in_waves", { jobId })
}

/** Cancel a running rank. */
export async function cancelRank(jobId: string): Promise<void> {
  return await appInvoke("cancel_rank", { jobId })
}

/** Get the sort report for a job. */
export async function getSortReport(jobId: string): Promise<SortReport> {
  return await appInvoke("get_sort_report", { jobId })
}
