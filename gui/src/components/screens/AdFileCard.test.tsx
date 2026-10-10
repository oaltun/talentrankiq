import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import { listen } from '@tauri-apps/api/event'
import { openPath } from '@tauri-apps/plugin-opener'
import { commands } from '@/lib/tauri-bindings'

import { AdFileCard } from './AdFileCard'
import { render } from '@/test/test-utils'

/**
 * The ad card mirrors a file the user edits in their *own* editor, so the only
 * thing that can tell it the content moved is the backend's `job-ad-reloaded`
 * event. This pins that wiring end to end at the component level — the Rust suite
 * proves serve *emits* the event, but nothing covered the gui acting on it, which
 * is exactly how a stale preview ships: edit, save, come back, still the old text.
 *
 * It also pins the single listener: the card invalidates its own query, so a
 * second listener would be a duplicate, not a safety net.
 */

// Not mocked globally in setup.ts (nothing else needs them): the OS file
// dialog has no gui in a test, and the opener is the point of the Open button.
vi.mock('@tauri-apps/plugin-opener', () => ({
  openPath: vi.fn().mockResolvedValue(undefined),
}))

const CHOSEN = '/Users/me/Documents/jobs/job-1/other-ad.txt'
vi.mock('@/lib/file-dialogs', () => ({
  openFileDialog: vi
    .fn()
    .mockResolvedValue('/Users/me/Documents/jobs/job-1/other-ad.txt'),
}))

type Handler = (event: { payload: unknown }) => void
const handlers = new Map<string, Handler>()

vi.mocked(listen).mockImplementation(async (event, handler) => {
  handlers.set(event, handler as unknown as Handler)
  return () => handlers.delete(event)
})

function emitJobAdReloaded(jobId: string) {
  const handler = handlers.get('job-ad-reloaded')
  if (!handler)
    throw new Error(
      'the ad card is not listening for job-ad-reloaded — a disk edit could never refresh the preview'
    )
  handler({ payload: { job_id: jobId } })
}

const PATH = '/Users/me/Documents/jobs/job-1/job-ad.txt'

/** The file's content as the backend would read it; tests move it along. */
let fileContent = 'Original advertisement.'
function mockFile() {
  vi.mocked(commands.getJobAd).mockImplementation(async () => ({
    status: 'ok',
    data: { path: PATH, text: fileContent, created: false },
  }))
}

/** The ad card, as it sits in the Setup pane. */
function renderCard() {
  return render(<AdFileCard jobId="job-1" />)
}

beforeEach(() => {
  handlers.clear()
  fileContent = 'Original advertisement.'
  mockFile()
  vi.mocked(openPath).mockClear()
  vi.mocked(commands.setJobAdPath).mockResolvedValue({
    status: 'ok',
    data: { path: PATH, text: fileContent, created: false },
  })
})

describe('AdFileCard', () => {
  it('shows the path and a read-only mirror of the file', async () => {
    renderCard()

    expect(await screen.findByText(PATH)).toBeInTheDocument()
    // Read-only mirror — a textarea whose value is the file text, read-only.
    const mirror = screen.getByRole('textbox', {
      name: /contents \(read-only\)/i,
    })
    expect(mirror).toHaveValue('Original advertisement.')
    expect(mirror).toHaveAttribute('readonly')
  })

  it('re-reads the file when the backend says it was reloaded', async () => {
    renderCard()
    expect(
      await screen.findByDisplayValue('Original advertisement.')
    ).toBeInTheDocument()

    // The user edited and saved in their own editor: the file on disk moved,
    // and the backend now reports different contents.
    fileContent = 'Edited in TextEdit and saved.'
    emitJobAdReloaded('job-1')

    await waitFor(() =>
      expect(
        screen.getByRole('textbox', { name: /contents \(read-only\)/i })
      ).toHaveValue('Edited in TextEdit and saved.')
    )
  })

  it('ignores another job’s reload event', async () => {
    renderCard()
    expect(
      await screen.findByDisplayValue('Original advertisement.')
    ).toBeInTheDocument()

    fileContent = 'A different job’s advertisement.'
    emitJobAdReloaded('job-2')

    // Give a wrong refetch a chance to land before asserting it did not.
    await waitFor(() =>
      expect(
        screen.getByRole('textbox', { name: /contents \(read-only\)/i })
      ).toHaveValue('Original advertisement.')
    )
  })

  it('opens the ad in the system editor', async () => {
    renderCard()
    await userEvent.click(await screen.findByRole('button', { name: /open/i }))

    expect(openPath).toHaveBeenCalledWith(PATH)
  })

  it('points the job at a different file', async () => {
    renderCard()
    await userEvent.click(
      await screen.findByRole('button', { name: /change/i })
    )

    expect(commands.setJobAdPath).toHaveBeenCalledWith('job-1', CHOSEN)
  })
})
