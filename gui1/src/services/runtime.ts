// Frontend service for runtime operations.

import { appInvoke } from "../lib/invoke"
import type { RuntimeStatusInfo, SystemBudget } from "../types"

/** Get the engine runtime status. */
export async function getRuntimeStatus(): Promise<RuntimeStatusInfo> {
  return await appInvoke("runtime_status")
}

/** Start the embedding engine runtime. */
export async function startRuntime(): Promise<void> {
  await appInvoke("start_runtime")
}

/** Get the system capacity budget. */
export async function getSystemBudget(): Promise<SystemBudget> {
  return await appInvoke("system_budget")
}
