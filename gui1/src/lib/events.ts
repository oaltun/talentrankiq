import { listen, type UnlistenFn } from "@tauri-apps/api/event"
import { isTauri } from "./invoke"

type EventCallback = (payload: any) => void

// Browser transport: one EventSource to the vite dev proxy, which forwards the
// CLI's event lines. In Tauri the native event system is used instead, so this
// module is the single place that knows how events reach the UI.
let source: EventSource | null = null
const listeners = new Map<string, Set<EventCallback>>()

function ensureSource(): EventSource {
  if (source) return source
  source = new EventSource("/api/events")
  source.onmessage = (event) => {
    let parsed: { event?: string; data?: unknown }
    try {
      parsed = JSON.parse(event.data)
    } catch {
      return
    }
    if (!parsed || typeof parsed.event !== "string") return
    const set = listeners.get(parsed.event)
    if (!set) return
    for (const cb of set) cb(parsed.data)
  }
  return source
}

/**
 * Subscribe to a CLI event by name. Returns an unlisten function.
 * Uses the native Tauri event system inside the app, and the dev-server SSE
 * bridge in the browser (Playwright e2e).
 */
export async function listenEvent<T = any>(
  name: string,
  callback: (payload: T) => void
): Promise<UnlistenFn> {
  if (isTauri()) {
    return await listen<T>(name, (event) => callback(event.payload))
  }

  let set = listeners.get(name)
  if (!set) {
    set = new Set()
    listeners.set(name, set)
  }
  set.add(callback as EventCallback)
  ensureSource()

  return async () => {
    set!.delete(callback as EventCallback)
  }
}
