import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mocks must be declared before the dynamic imports below so every test
// imports a freshly-initialised module (file-dialogs caches the fallback dir).
vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn(),
  save: vi.fn(),
}))

vi.mock('@tauri-apps/api/path', () => ({
  documentDir: vi.fn(),
  homeDir: vi.fn(),
}))

import { open, save } from '@tauri-apps/plugin-dialog'
import { documentDir, homeDir } from '@tauri-apps/api/path'
import type { Mock } from 'vitest'

const openMock = open as unknown as Mock
const saveMock = save as unknown as Mock
const documentDirMock = documentDir as unknown as Mock
const homeDirMock = homeDir as unknown as Mock

/** Import a fresh module instance so last-dir/fallback caches are empty. */
async function fresh() {
  return await import('./file-dialogs')
}

describe('file-dialogs', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.resetModules()
    localStorage.clear()
    documentDirMock.mockResolvedValue('/Users/apple1/Documents')
    homeDirMock.mockResolvedValue('/home/o')
    openMock.mockResolvedValue(null)
    saveMock.mockResolvedValue(null)
  })

  it('opens at the OS Documents folder when nothing was remembered', async () => {
    const { openFileDialog } = await fresh()
    await openFileDialog()
    expect(openMock).toHaveBeenCalledWith(
      expect.objectContaining({ defaultPath: '/Users/apple1/Documents' })
    )
  })

  it('falls back to the home directory when Documents is unavailable', async () => {
    documentDirMock.mockRejectedValue(new Error('no xdg user dirs'))
    const { openFileDialog } = await fresh()
    await openFileDialog()
    expect(openMock).toHaveBeenCalledWith(
      expect.objectContaining({ defaultPath: '/home/o' })
    )
  })

  it('remembers a picked directory itself, not its parent', async () => {
    const { openFileDialog } = await fresh()
    openMock.mockResolvedValue('/Users/apple1/cvs')
    await openFileDialog({ directory: true })
    await openFileDialog()
    expect(openMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ defaultPath: '/Users/apple1/cvs' })
    )
  })

  it('remembers the directory of a picked file for the next dialog', async () => {
    const { openFileDialog } = await fresh()
    openMock.mockResolvedValue('/Users/apple1/cvs/jane.pdf')
    await openFileDialog()
    await openFileDialog()
    expect(openMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ defaultPath: '/Users/apple1/cvs' })
    )
  })

  it('an explicit startDir wins over the remembered directory', async () => {
    const { openFileDialog } = await fresh()
    openMock.mockResolvedValue('/Users/apple1/cvs/jane.pdf')
    await openFileDialog()
    await openFileDialog({ startDir: '/srv/jobs' })
    expect(openMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ defaultPath: '/srv/jobs' })
    )
  })

  it('a cancelled dialog does not change the remembered directory', async () => {
    const { openFileDialog } = await fresh()
    openMock.mockResolvedValue('/Users/apple1/cvs/jane.pdf')
    await openFileDialog()
    openMock.mockResolvedValue(null)
    await openFileDialog()
    openMock.mockResolvedValue(undefined)
    await openFileDialog()
    expect(openMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ defaultPath: '/Users/apple1/cvs' })
    )
  })

  it('saves into the remembered directory when given a bare file name', async () => {
    const { saveFileDialog, openFileDialog } = await fresh()
    openMock.mockResolvedValue('/Users/apple1/exports')
    await openFileDialog({ directory: true })
    saveMock.mockResolvedValue('/Users/apple1/exports/triqui.csv')
    await saveFileDialog({ defaultPath: 'triqui.csv' })
    expect(saveMock).toHaveBeenCalledWith(
      expect.objectContaining({
        defaultPath: '/Users/apple1/exports/triqui.csv',
      })
    )
  })

  it('leaves save default paths containing a directory untouched', async () => {
    const { saveFileDialog } = await fresh()
    await saveFileDialog({ defaultPath: '/tmp/out/triqui.csv' })
    expect(saveMock).toHaveBeenCalledWith(
      expect.objectContaining({ defaultPath: '/tmp/out/triqui.csv' })
    )
  })

  it('remembers the directory of a saved file across save/open calls', async () => {
    const { saveFileDialog, openFileDialog } = await fresh()
    saveMock.mockResolvedValue('/Users/apple1/exports/bundle.triquijob')
    await saveFileDialog()
    await openFileDialog()
    expect(openMock).toHaveBeenCalledWith(
      expect.objectContaining({ defaultPath: '/Users/apple1/exports' })
    )
  })
})
