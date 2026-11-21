// Frontend service for criteria management.

import { appInvoke } from "../lib/invoke"
import type { Criterion } from "../types"

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
  return await appInvoke("delete_criterion", { id })
}
