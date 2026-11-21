import { invoke } from "@tauri-apps/api/core"

export function isTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    ("__TAURI__" in window || "__TAURI_INTERNALS__" in window)
  )
}

/**
 * Tauri's `invoke` maps JS camelCase argument names to the Rust command's
 * snake_case parameter names. The dev proxy talks to the CLI protocol
 * directly, so apply the same mapping here or the CLI rejects the request
 * (`unknown field 'jobId', expected 'job_id'`).
 */
function toSnakeCaseParams(
  params: Record<string, unknown>
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(params)) {
    out[key.replace(/([A-Z])/g, "_$1").toLowerCase()] = value
  }
  return out
}

export async function browserInvoke(
  cmd: string,
  params: Record<string, unknown> = {}
): Promise<any> {
  const t0 = performance.now()
  console.log(`[invoke] → ${cmd}`)
  try {
    const res = await fetch("/api/invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd, params: toSnakeCaseParams(params) }),
    })
    const json = await res.json().catch(() => ({}))
    if (!res.ok)
      throw new Error(json?.err?.message || json?.error || `HTTP ${res.status}`)
    if (json && json.err)
      throw new Error(json.err.message || json.err.code || "triq error")
    const ok =
      json && Object.prototype.hasOwnProperty.call(json, "ok") ? json.ok : json
    console.log(`[invoke] ← ${cmd} ${Math.round(performance.now() - t0)}ms`)
    return ok
  } catch (e: any) {
    console.log(
      `[invoke] ✘ ${cmd} ${Math.round(performance.now() - t0)}ms: ${e?.message || e}`
    )
    throw e
  }
}

export async function appInvoke<T = any>(
  cmd: string,
  params?: Record<string, unknown>
): Promise<T> {
  if (isTauri()) {
    const t0 = performance.now()
    console.log(`[invoke] → ${cmd}`)
    try {
      // The Rust shell exposes a single dispatcher that forwards to the same
      // `triq serve` process the dev proxy uses, so protocol behavior (params,
      // errors, events) is identical in both transports.
      const out = await invoke<T>("triq_dispatch", {
        cmd,
        params: params ?? {},
      })
      console.log(`[invoke] ← ${cmd} ${Math.round(performance.now() - t0)}ms`)
      return out
    } catch (e: any) {
      console.log(
        `[invoke] ✘ ${cmd} ${Math.round(performance.now() - t0)}ms: ${e?.message || e}`
      )
      throw e
    }
  }
  return (await browserInvoke(cmd, params ?? {})) as T
}
