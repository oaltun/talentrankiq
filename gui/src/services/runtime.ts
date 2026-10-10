import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { useQuery, useQueryClient } from '@tanstack/react-query'

import { commands, unwrapResult } from '@/lib/tauri-bindings'
import type { RuntimeStatusInfo } from '@/lib/tauri-bindings'

export const runtimeQueryKeys = {
  all: ['runtime'] as const,
  status: () => [...runtimeQueryKeys.all, 'status'] as const,
}

/**
 * Engine status query. Kept fresh by `useRuntimeStatusListener` (mounted once in the
 * shell), which invalidates on `runtime-status` events emitted from Rust.
 */
export function useRuntimeStatus() {
  return useQuery({
    queryKey: runtimeQueryKeys.status(),
    queryFn: async (): Promise<RuntimeStatusInfo> =>
      unwrapResult(await commands.runtimeStatus()),
  })
}

/** True when the embedding engine is up and ready to embed/rank. */
export function runtimeReady(s: RuntimeStatusInfo | undefined): boolean {
  return !!s && s.status === 'Running'
}

/**
 * Subscribe once (mount in the shell) to `runtime-status` events and refresh the
 * status query. The payload mirrors `RuntimeStatusInfo` but we only re-fetch.
 */
export function useRuntimeStatusListener() {
  const qc = useQueryClient()
  useEffect(() => {
    let mounted = true
    let unlisten: (() => void) | null = null

    listen('runtime-status', () => {
      qc.invalidateQueries({ queryKey: runtimeQueryKeys.status() })
    })
      .then(fn => {
        if (mounted) unlisten = fn
        else fn()
      })
      .catch(error => {
        console.error('Failed to listen for runtime status', error)
      })

    return () => {
      mounted = false
      if (unlisten) unlisten()
    }
  }, [qc])
}
