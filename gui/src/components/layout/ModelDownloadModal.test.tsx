import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'

import { listen } from '@tauri-apps/api/event'
import { commands } from '@/lib/tauri-bindings'

import { ModelDownloadModal } from './ModelDownloadModal'
import { render } from '@/test/test-utils'

// The modal is wired to the SAME pieces as the Models activity: the
// `downloadEmbedder` command (one shared download path) and the
// `model-download-progress` events. These tests pin that wiring.

type ProgressHandler = (event: { payload: unknown }) => void
const progressHandlers = new Map<string, ProgressHandler>()

vi.mocked(listen).mockImplementation(async (event, handler) => {
  progressHandlers.set(event, handler as unknown as ProgressHandler)
  return () => progressHandlers.delete(event)
})

function emitModelDownloadProgress(payload: {
  repo_id: string
  filename: string
  done: number
  total: number | null
}) {
  const handler = progressHandlers.get('model-download-progress')
  if (!handler) throw new Error('modal is not listening for progress events')
  handler({ payload })
}

const embedder = {
  slug: 'test-model-q4',
  name: 'Test Model',
  memory_mb: 600,
  downloaded: false,
  is_default: true,
  dim: 640,
  max_context_window: 32768,
  source: 'hf:test/test-model-ONNX',
  size_bytes: 0,
}

beforeEach(() => {
  vi.mocked(commands.listEmbedders)
    .mockClear()
    .mockResolvedValue({
      status: 'ok',
      data: [embedder],
    })
  vi.mocked(commands.downloadEmbedder).mockClear().mockResolvedValue({
    status: 'ok',
    data: null,
  })
  vi.mocked(commands.modelVariants)
    .mockClear()
    .mockResolvedValue({
      status: 'ok',
      data: {
        model: 'test-model-q4',
        runtime: 'ort',
        present: false,
        onnx_present: false,
        mlx_present: null,
      },
    })
  vi.mocked(commands.startRuntime).mockClear()
})

describe('ModelDownloadModal', () => {
  it('asks for consent first, then drives the progress bar from events', async () => {
    let finishDownload: (() => void) | undefined
    vi.mocked(commands.downloadEmbedder).mockImplementation(
      () =>
        new Promise(resolve => {
          finishDownload = () => resolve({ status: 'ok', data: null })
        })
    )

    render(<ModelDownloadModal />)

    // Missing arm → consent question, NO download yet, engine not started.
    expect(
      await screen.findByText(/variant of Test Model/i)
    ).toBeInTheDocument()
    expect(commands.downloadEmbedder).not.toHaveBeenCalled()
    expect(commands.startRuntime).not.toHaveBeenCalled()

    // Consent starts the download through the SAME command the Models
    // activity uses.
    await userEvent.click(screen.getByRole('button', { name: /download/i }))
    await waitFor(() => {
      expect(commands.downloadEmbedder).toHaveBeenCalledWith('test-model-q4')
    })
    expect(await screen.findByText('Downloading…')).toBeInTheDocument()

    // No totals known yet → indeterminate pulse placeholder, no percent bar.
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()

    // Progress events (aggregate bytes, exactly what serve now emits) move
    // the real bar.
    emitModelDownloadProgress({
      repo_id: 'test-model-q4',
      filename: 'model_q4f16.onnx',
      done: 300,
      total: 600,
    })
    const bar = await screen.findByRole('progressbar')
    expect(bar.getAttribute('aria-valuenow')).toBe('50')

    emitModelDownloadProgress({
      repo_id: 'test-model-q4',
      filename: 'model_q4f16.onnx',
      done: 600,
      total: 600,
    })
    await waitFor(() => {
      expect(
        screen.getByRole('progressbar').getAttribute('aria-valuenow')
      ).toBe('100')
    })

    // Download settles → model on disk → engine starts, modal goes away.
    finishDownload?.()
    await waitFor(() => {
      expect(commands.startRuntime).toHaveBeenCalled()
      expect(screen.queryByText('Downloading…')).not.toBeInTheDocument()
    })
  })

  it('starts nothing when the user declines', async () => {
    render(<ModelDownloadModal />)
    expect(
      await screen.findByText(/variant of Test Model/i)
    ).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /not now/i }))
    await waitFor(() => {
      expect(
        screen.queryByText(/variant of Test Model/i)
      ).not.toBeInTheDocument()
    })
    expect(commands.downloadEmbedder).not.toHaveBeenCalled()
    expect(commands.startRuntime).not.toHaveBeenCalled()
  })

  it('shows the failure and retries through the same shared download', async () => {
    vi.mocked(commands.downloadEmbedder).mockRejectedValueOnce(
      'disk full (exit 3)'
    )

    render(<ModelDownloadModal />)
    // Consent first…
    await userEvent.click(
      await screen.findByRole('button', { name: /download/i })
    )
    expect(await screen.findByText('disk full (exit 3)')).toBeInTheDocument()

    vi.mocked(commands.downloadEmbedder).mockResolvedValue({
      status: 'ok',
      data: null,
    })
    await userEvent.click(screen.getByRole('button', { name: /retry/i }))
    await waitFor(() => {
      expect(commands.downloadEmbedder).toHaveBeenCalledTimes(2)
    })
  })

  it('does not appear when the effective arm is already on disk', async () => {
    vi.mocked(commands.modelVariants).mockResolvedValue({
      status: 'ok',
      data: {
        model: 'test-model-q4',
        runtime: 'ort',
        present: true,
        onnx_present: true,
        mlx_present: null,
      },
    })
    render(<ModelDownloadModal />)
    await waitFor(() => {
      expect(commands.startRuntime).toHaveBeenCalled()
    })
    expect(commands.downloadEmbedder).not.toHaveBeenCalled()
    expect(screen.queryByText(/variant of Test Model/i)).not.toBeInTheDocument()
  })
})
