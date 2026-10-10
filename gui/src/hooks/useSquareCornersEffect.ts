import { useEffect } from 'react'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { useUIStore } from '@/store/ui-store'
import { usePlatform } from './use-platform'

/**
 * Manages square corners based on platform and fullscreen state.
 *
 * Rules:
 * - macOS: always square (native decorations draw the rounded window frame,
 *   so the content area underneath is a plain rectangle)
 * - Windows: square when fullscreen (no rounded corners needed at screen edge)
 * - Linux: square when fullscreen
 */
export function useSquareCornersEffect() {
  const platform = usePlatform()
  const setSquareCorners = useUIStore(state => state.setSquareCorners)

  useEffect(() => {
    // macOS uses native window decorations; the OS rounds the frame itself.
    if (platform === 'macos') {
      setSquareCorners(true)
      return
    }

    let cancelled = false
    const window = getCurrentWindow()

    const updateCorners = async () => {
      const isFullscreen = await window.isFullscreen()
      if (cancelled) return
      // Windows/Linux: square corners only in fullscreen
      setSquareCorners(isFullscreen)
    }

    // Check initial state
    void updateCorners()

    // Listen for window state changes
    const unlisten = window.onResized(() => {
      if (cancelled) return
      void updateCorners()
    })

    return () => {
      cancelled = true
      void unlisten.then(fn => fn())
    }
  }, [platform, setSquareCorners])
}
