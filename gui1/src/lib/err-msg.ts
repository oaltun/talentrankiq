// Tauri commands declared as `Result<T, String>` reject the JS promise with a
// plain string, not an Error instance, so `e instanceof Error` misses them and
// the backend's message (e.g. the comparator filename gate) is lost.
export function errMsg(e: unknown, fallback: string): string {
  if (e instanceof Error) return e.message
  if (typeof e === "string" && e.length > 0) return e
  return fallback
}
