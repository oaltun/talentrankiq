// Frontend service for results CSV export.

import { appInvoke } from "../lib/invoke"
import type { ResultsCsvSummary } from "../types"

/** Export ranked results as CSV. */
export async function exportResultsCsv(
  jobId: string,
  outPath: string
): Promise<ResultsCsvSummary> {
  return await appInvoke("export_results_csv", { jobId, outPath })
}
