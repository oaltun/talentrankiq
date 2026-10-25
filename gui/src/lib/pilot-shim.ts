// Pilot JS Shim - replaces broken tauri-pilot JS bridge injection
// Listens for Tauri events from Rust, executes scripts, returns results via callback command

import { listen } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/core'

interface PilotQueryResult {
  tag: string
  id: string | null
  classes: string[]
  text: string
  rect: { x: number; y: number; width: number; height: number } | null
  attributes: Record<string, string>
  children: unknown[]
}

interface PilotSnapshotResult {
  url: string
  title: string
  elements: PilotQueryResult[]
}

interface PilotClickResult {
  success: boolean
  error?: string
}

interface PilotExecuteEvent {
  id: number
  script: string
}

type EventCallback = (payload: unknown) => void

class PilotShim {
  private unlisten: (() => void) | null = null
  private eventListeners: Map<string, Set<EventCallback>> = new Map<
    string,
    Set<EventCallback>
  >()
  private eventUnlistenIds: Map<string, number> = new Map<string, number>()
  private eventIdCounter = 1

  constructor() {
    if (typeof window !== 'undefined') {
      this.init()
    }
  }

  private async init() {
    // Listen for execute events from Rust
    this.unlisten = await listen<PilotExecuteEvent>('pilot:execute', event => {
      this.handleExecute(event.payload)
    })

    // Listen for forwarded events from Rust (pilot:event:*)
    const wildcardUnlisten = await listen('pilot:event:*', event => {
      const eventName = event.event.replace('pilot:event:', '')
      const listeners = this.eventListeners.get(eventName)
      if (listeners) {
        for (const cb of listeners) {
          cb(event.payload)
        }
      }
    })
    // Store wildcard unlisten for cleanup
    this.wildcardUnlisten = wildcardUnlisten

    // Signal readiness to Rust side
    try {
      await this.invokePilotBridge({ id: 0, request: { type: 'ready' } })
      console.log('[PilotShim] Ready')
    } catch (e) {
      console.warn('[PilotShim] Init failed:', e)
    }
  }

  private wildcardUnlisten: (() => void) | null = null

  private async handleExecute(payload: PilotExecuteEvent) {
    try {
      // Execute the script in the page context
      const result = eval(payload.script)

      // Handle promises (async scripts)
      const resolvedResult = result instanceof Promise ? await result : result

      // Send result back via callback command
      await invoke('pilot_callback', {
        id: payload.id,
        result: resolvedResult ?? null,
        error: undefined,
      })
    } catch (e) {
      // Send error back via callback command
      await invoke('pilot_callback', {
        id: payload.id,
        result: null,
        error: e instanceof Error ? e.message : String(e),
      })
    }
  }

  private async invokePilotBridge(message: object): Promise<string> {
    return invoke('pilot_bridge', { message: JSON.stringify(message) })
  }

  // Query elements by selector, return serialized a11y-tree-ish data
  async query(selector: string): Promise<PilotQueryResult[]> {
    const response = await this.invokePilotBridge({
      id: Date.now(),
      request: { type: 'query', selector },
    })
    return JSON.parse(response).result
  }

  // Click element by selector
  async click(selector: string): Promise<PilotClickResult> {
    const response = await this.invokePilotBridge({
      id: Date.now(),
      request: { type: 'click', selector },
    })
    return JSON.parse(response).result
  }

  // Evaluate arbitrary JavaScript
  async eval(js: string): Promise<unknown> {
    const response = await this.invokePilotBridge({
      id: Date.now(),
      request: { type: 'eval', js },
    })
    return JSON.parse(response).result
  }

  // Snapshot the entire DOM as a11y-tree-ish structure
  async snapshot(): Promise<PilotSnapshotResult> {
    const response = await this.invokePilotBridge({
      id: Date.now(),
      request: { type: 'snapshot' },
    })
    return JSON.parse(response).result
  }

  // Listen for an event (uses Rust-side listener via pilot bridge)
  async on(event: string, callback: EventCallback): Promise<void> {
    const listeners = this.eventListeners.get(event) || new Set()
    listeners.add(callback)
    this.eventListeners.set(event, listeners)

    // If this is the first listener for this event, register with Rust
    const isFirstListener = listeners.size === 1
    if (isFirstListener) {
      const id = this.eventIdCounter++
      try {
        await invoke('event_listen', { event, id })
        this.eventUnlistenIds.set(event, id)
      } catch (e) {
        console.error(`[PilotShim] Failed to listen for event ${event}:`, e)
        listeners.delete(callback)
        if ((listeners.size as number) === 0) {
          this.eventListeners.delete(event)
        }
        throw e
      }
    }
  }

  // Stop listening for an event
  async off(event: string, callback: EventCallback): Promise<void> {
    const listeners = this.eventListeners.get(event)
    if (!listeners) return

    listeners.delete(callback)
    if (listeners.size === 0) {
      this.eventListeners.delete(event)
      const id = this.eventUnlistenIds.get(event)
      if (id !== undefined) {
        try {
          await invoke('event_unlisten', { id })
        } catch (e) {
          console.error(`[PilotShim] Failed to unlisten for event ${event}:`, e)
        }
        this.eventUnlistenIds.delete(event)
      }
    }
  }

  // Cleanup
  destroy() {
    if (this.unlisten) {
      this.unlisten()
      this.unlisten = null
    }
    if (this.wildcardUnlisten) {
      this.wildcardUnlisten()
      this.wildcardUnlisten = null
    }
    // Unregister all events
    for (const [event, id] of this.eventUnlistenIds.entries()) {
      invoke('event_unlisten', { id }).catch(e => {
        console.error(`[PilotShim] Failed to unlisten for event ${event}:`, e)
      })
    }
    this.eventListeners.clear()
    this.eventUnlistenIds.clear()
  }
}

// Export singleton
export const pilotShim = new PilotShim()

// Also expose globally for console debugging
if (typeof window !== 'undefined') {
  ;(window as Record<string, unknown>).__PILOT_SHIM__ = pilotShim
}
