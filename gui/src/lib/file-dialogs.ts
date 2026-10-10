/**
 * Single entry point for native open/save dialogs so every file picker in the
 * app starts in the same place: the directory the user last picked anything
 * from (persisted across sessions), falling back to the OS Documents folder,
 * then the user's home directory. All call sites must go through here instead
 * of calling the dialog plugin directly.
 */
import { open, save } from '@tauri-apps/plugin-dialog'
import { documentDir, homeDir } from '@tauri-apps/api/path'
import { logger } from '@/lib/logger'

const LAST_DIR_KEY = 'file-dialog-last-dir'

/** Cached OS fallback directory (Documents, then home). */
let fallbackCache: string | undefined

async function fallbackDirectory(): Promise<string | undefined> {
  if (fallbackCache) return fallbackCache
  for (const resolve of [documentDir, homeDir]) {
    try {
      const dir = await resolve()
      if (dir) {
        fallbackCache = dir
        return dir
      }
    } catch (error) {
      logger.warn('Could not resolve fallback dialog directory', { error })
    }
  }
  return undefined
}

function storedLastDir(): string | undefined {
  try {
    return localStorage.getItem(LAST_DIR_KEY) ?? undefined
  } catch {
    return undefined
  }
}

/** Remember the directory a selection came from (the path itself for dirs). */
function rememberDirectoryOf(path: string, isDirectory: boolean): void {
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))
  const dir = !isDirectory && cut > 0 ? path.slice(0, cut) : path
  try {
    localStorage.setItem(LAST_DIR_KEY, dir)
  } catch {
    // Persistence is best-effort; the in-session default still works.
  }
}

export interface FileDialogFilter {
  name: string
  extensions: string[]
}

export interface OpenFileDialogOptions {
  title?: string
  filters?: FileDialogFilter[]
  multiple?: boolean
  directory?: boolean
  /**
   * Preferred start directory (e.g. the job's stored CV dir). When omitted the
   * shared last-used directory is used, then the OS fallback.
   */
  startDir?: string
}

export async function openFileDialog(
  options: OpenFileDialogOptions = {}
): Promise<string | string[] | null> {
  const defaultPath =
    options.startDir || storedLastDir() || (await fallbackDirectory())
  const selection = await open({ ...options, defaultPath })
  const first = typeof selection === 'string' ? selection : selection?.[0]
  if (first) rememberDirectoryOf(first, !!options.directory)
  return selection ?? null
}

export interface SaveFileDialogOptions {
  title?: string
  filters?: FileDialogFilter[]
  /** Suggested file name (no directory) — joined onto the shared start dir. */
  defaultPath?: string
}

export async function saveFileDialog(
  options: SaveFileDialogOptions = {}
): Promise<string | null> {
  const dir = storedLastDir() || (await fallbackDirectory())
  const suggested = options.defaultPath
  const defaultPath =
    suggested && dir && !suggested.includes('/') && !suggested.includes('\\')
      ? `${dir}/${suggested}`
      : suggested || dir
  const path = await save({ ...options, defaultPath })
  if (path) rememberDirectoryOf(path, false)
  return path ?? null
}
