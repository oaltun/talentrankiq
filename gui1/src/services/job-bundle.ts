// Frontend service for job bundle export/import.

import { appInvoke } from "../lib/invoke"
import type { BundleExportOptions, BundleExportSummary } from "../types"

/** Export job as portable .triquijob bundle. */
export async function exportJobBundle(
  jobId: string,
  options: BundleExportOptions,
  outPath: string
): Promise<BundleExportSummary> {
  return await appInvoke("export_job_bundle", {
    jobId,
    options,
    outPath,
  })
}
