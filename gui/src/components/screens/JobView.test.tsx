import { screen, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import { commands } from '@/lib/tauri-bindings'

import { JobView } from './JobView'
import { render } from '@/test/test-utils'

/**
 * The job ad is a file the user owns, so it lives in the Setup pane: its AD FILE
 * card and a read-only peek at the contents. There is no pane of its own — a
 * third column bought two buttons and a mirror a permanent share of the width
 * that the candidate table needs.
 *
 * This pins where the ad's controls live, because the tempting regression is
 * "let's give the ad its own pane again" — which is exactly what this workspace
 * deliberately does not have.
 */

// Routing is not what this test is about; JobView just needs its params.
vi.mock('@tanstack/react-router', () => ({
  useParams: () => ({ jobId: 'job-1' }),
  useNavigate: () => vi.fn(),
}))

const CONTENTS_TEXTBOX = /contents \(read-only\)/i
const PATH = '/tmp/jobs/job-1/job-ad.txt'

/**
 * The pane whose heading is `name`, so "which pane is this in?" is assertable.
 * Async: the workspace renders a spinner until the job query resolves, so the
 * heading is not in the document on the first tick.
 */
async function pane(name: RegExp) {
  const heading = await screen.findByRole('heading', { name })
  return heading.closest('section') as HTMLElement
}

beforeEach(() => {
  vi.mocked(commands.loadPreferences).mockResolvedValue({
    status: 'ok',
    data: { theme: 'system' },
  } as never)
  vi.mocked(commands.getJobAd).mockResolvedValue({
    status: 'ok',
    data: { path: PATH, text: 'Original advertisement.', created: false },
  })
})

describe('JobView ad placement', () => {
  it('puts the ad file card and its contents mirror in the Setup pane', async () => {
    render(<JobView />)

    const setup = await pane(/setup/i)
    expect(await within(setup).findByText('Ad file')).toBeInTheDocument()
    expect(within(setup).getByText(PATH)).toBeInTheDocument()
    expect(
      await within(setup).findByRole('textbox', { name: CONTENTS_TEXTBOX })
    ).toHaveValue('Original advertisement.')
  })

  it('has no separate job-ad pane', async () => {
    render(<JobView />)

    // The ad's controls are in Setup, and only in Setup — a second copy would be
    // two live sets of the same controls.
    await screen.findByRole('heading', { name: 'Setup' })
    expect(
      screen.queryByRole('heading', { name: /^job ad$/i })
    ).not.toBeInTheDocument()
    const setup = await pane(/setup/i)
    expect(within(setup).getAllByText('Ad file')).toHaveLength(1)
  })
})
