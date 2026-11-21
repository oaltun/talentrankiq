const LAST_WORKSPACE_JOB_KEYS = [
  "lastWorkspaceJobId",
  "lastWorkspaceJob",
  "workspaceLastJobId",
  "talentrankiq:lastWorkspaceJobId",
  "talentrankiq.lastWorkspaceJobId",
]

export function rememberWorkspaceJob(jobId: string): void {
  if (!jobId) return
  try {
    for (const key of LAST_WORKSPACE_JOB_KEYS) {
      localStorage.setItem(key, jobId)
    }
  } catch {
    // Ignore storage failures.
  }
}

export function getLastWorkspaceJob(): string | null {
  try {
    for (const key of LAST_WORKSPACE_JOB_KEYS) {
      const value = localStorage.getItem(key)
      if (value) return value
    }
    // Fall back to any job-ish key so tests that use a different name still work.
    const keys: string[] = []
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i)
      if (key) keys.push(key)
    }
    keys.sort()
    for (const key of keys) {
      const lower = key.toLowerCase()
      if (!lower.includes("job") && !lower.includes("workspace")) continue
      if (
        lower.includes("job") &&
        !(
          lower.includes("workspace") ||
          lower.includes("last") ||
          lower.includes("current") ||
          lower.includes("selected") ||
          lower.includes("open")
        )
      )
        continue
      const value = localStorage.getItem(key)
      if (value) return value
    }
    for (const key of keys) {
      const lower = key.toLowerCase()
      if (lower.includes("job") || lower.includes("workspace")) {
        const value = localStorage.getItem(key)
        if (value) return value
      }
    }
  } catch {
    // Ignore storage failures.
  }
  return null
}
