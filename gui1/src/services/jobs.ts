// Frontend service for job management.

import { appInvoke } from "../lib/invoke"
import type {
  Job,
  JobSummary,
  JobDetail,
  JobAdFile,
  Criterion,
  BundleExportOptions,
  BundleExportSummary,
  ResultsCsvSummary,
} from "../types"

export type {
  Job,
  JobSummary,
  JobDetail,
  JobAdFile,
  Criterion,
  BundleExportOptions,
  BundleExportSummary,
  ResultsCsvSummary,
}

/** List all job summaries (active + archived) from the triq serve loop. */
export async function listJobs(): Promise<JobSummary[]> {
  return await appInvoke("list_job_summaries")
}

/** Create a new empty job (auto-titled `job-N`) and return its full Job object. */
export async function createJob(): Promise<Job> {
  return await appInvoke("create_job")
}

/** Archive or restore a job by id. */
export async function archiveJob(
  jobId: string,
  archived: boolean
): Promise<void> {
  await appInvoke("archive_job", { jobId, archived })
}

/** Delete a job by id (candidates are cascade-deleted by the DB). */
export async function deleteJob(jobId: string): Promise<void> {
  await appInvoke("delete_job", { jobId })
}

/**
 * Make `job_id` the only watched CV directory: serve drops all other
 * watchers, creates the job's CV folder when missing, and runs a catch-up
 * import scan.
 */
export async function watchJob(jobId: string): Promise<Job> {
  return await appInvoke("watch_job", { jobId })
}

/** Get full job detail including criteria, rubric, triqui, and counts. */
export async function getJobDetail(jobId: string): Promise<JobDetail> {
  return await appInvoke("get_job_detail", { jobId })
}

/** Get the job-ad file (path + contents) for a job. */
export async function getJobAd(jobId: string): Promise<JobAdFile> {
  return await appInvoke("get_job_ad", { jobId })
}

/** Change the job-ad file path for a job. */
export async function setJobAdPath(
  jobId: string,
  path: string
): Promise<JobAdFile> {
  return await appInvoke("set_job_ad_path", { jobId, path })
}

/** Re-read the job-ad text from disk. */
export async function reloadJobAd(jobId: string): Promise<JobAdFile> {
  return await appInvoke("reload_job_ad", { jobId })
}

/** Trigger CV import for a job's watched directory. */
export async function importCvDirectory(
  jobId: string,
  directory: string
): Promise<void> {
  await appInvoke("import_cv_directory", { jobId, directory })
}
/** Set the CV directory for a job (and optionally make it the only watched one). */
export async function setJobCvDir(
  jobId: string,
  directory: string,
  options?: { watch?: boolean }
): Promise<Job> {
  return await appInvoke("set_job_cv_dir", {
    jobId,
    directory,
    watch: options?.watch,
  })
}

/** Update job title, ad text, and/or triqui size. */
export async function updateJob(
  jobId: string,
  title: string,
  jobAdText: string | null,
  triquiSize: number
): Promise<Job> {
  return await appInvoke("update_job", { jobId, title, jobAdText, triquiSize })
}

/** Update job rubric weights. */
export async function updateJobWeights(
  jobId: string,
  adWeight: number,
  fbJobWeight: number,
  fbProfileWeight: number
): Promise<Job> {
  return await appInvoke("update_job_weights", {
    jobId,
    adWeight,
    fbJobWeight,
    fbProfileWeight,
  })
}

/** Add a criterion to a job. */
export async function addCriterion(
  jobId: string,
  text: string,
  weight?: number
): Promise<Criterion> {
  return await appInvoke("add_criterion", { jobId, text, weight })
}

/** Update a criterion. */
export async function updateCriterion(
  id: string,
  text: string,
  weight: number
): Promise<Criterion> {
  return await appInvoke("update_criterion", { id, text, weight })
}

/** Delete a criterion. */
export async function deleteCriterion(id: string): Promise<void> {
  await appInvoke("delete_criterion", { id })
}

/** Clear all CVs for a job. */
export async function emptyCandidates(jobId: string): Promise<void> {
  await appInvoke("empty_candidates", { jobId })
}

/** Export ranked results as CSV. */
export async function exportResultsCsv(
  jobId: string,
  outPath: string
): Promise<ResultsCsvSummary> {
  return await appInvoke("export_results_csv", { jobId, outPath })
}

/** Export job as portable .triquijob bundle. */
export async function exportJobBundle(
  jobId: string,
  options: BundleExportOptions,
  outPath: string
): Promise<BundleExportSummary> {
  return await appInvoke("export_job_bundle", { jobId, options, outPath })
}
