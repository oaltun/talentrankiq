import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Node 22+ ships experimental `localStorage`/`sessionStorage` globals that are
// no-op getters (undefined) unless node runs with `--localstorage-file`.
// vitest's jsdom environment skips copying any window key that is already
// `in global`, so jsdom's working Storage never lands on globalThis and every
// bare `localStorage` reference hits Node's broken getter. Copy the real
// jsdom storage back onto the global when the shadow is broken.
for (const key of ['localStorage', 'sessionStorage'] as const) {
  if (typeof globalThis[key] === 'undefined') {
    const storage = (
      globalThis as unknown as { jsdom?: { window?: Record<string, unknown> } }
    ).jsdom?.window?.[key]
    if (storage) {
      Object.defineProperty(globalThis, key, {
        value: storage,
        configurable: true,
        writable: true,
      })
    }
  }
}

// Mock matchMedia for tests
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock Tauri APIs for tests
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn().mockResolvedValue(() => {
    // Mock unlisten function
  }),
}))

// Window API — getCurrentWindow() reads Tauri internals unavailable in jsdom.
vi.mock('@tauri-apps/api/window', () => ({
  getCurrentWindow: () => ({
    isMaximized: vi.fn().mockResolvedValue(false),
    onResized: vi.fn().mockResolvedValue(() => {
      // mock unlisten
    }),
    maximize: vi.fn().mockResolvedValue(undefined),
    unmaximize: vi.fn().mockResolvedValue(undefined),
    setFocus: vi.fn().mockResolvedValue(undefined),
    show: vi.fn().mockResolvedValue(undefined),
    hide: vi.fn().mockResolvedValue(undefined),
    isVisible: vi.fn().mockResolvedValue(true),
  }),
}))

vi.mock('@tauri-apps/plugin-updater', () => ({
  check: vi.fn().mockResolvedValue(null),
}))

// Mock typed Tauri bindings (tauri-specta generated)
vi.mock('@/lib/tauri-bindings', () => ({
  commands: {
    greet: vi.fn().mockResolvedValue('Hello, test!'),
    loadPreferences: vi.fn().mockResolvedValue({
      status: 'ok',
      data: { theme: 'system' },
    }),
    savePreferences: vi.fn().mockResolvedValue({ status: 'ok', data: null }),
    sendNativeNotification: vi
      .fn()
      .mockResolvedValue({ status: 'ok', data: null }),
    saveEmergencyData: vi.fn().mockResolvedValue({ status: 'ok', data: null }),
    loadEmergencyData: vi.fn().mockResolvedValue({ status: 'ok', data: null }),
    cleanupOldRecoveryFiles: vi
      .fn()
      .mockResolvedValue({ status: 'ok', data: 0 }),
    listJobSummaries: vi.fn().mockResolvedValue({ status: 'ok', data: [] }),
    getJobDetail: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        job: {
          id: 'job-1',
          title: 'Job 1',
          job_ad_text: 'Sample job advertisement.',
          import_directory_path: '/tmp/jobs/job-1/cv',
          triqui_size: 20,
          archived: false,
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
          last_ranked_at: null,
        },
        criteria: [],
        rubric: [],
        triqui: [],
        cv_imported: 0,
        cv_failed: 0,
        importing: false,
      },
    }),
    logStdout: vi.fn().mockResolvedValue(undefined),
    startRuntime: vi.fn().mockResolvedValue(undefined),
    jobUiState: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        engine_ready: true,
        sort_allowed: true,
        rerank_dirty: false,
        reason: 'clean',
        rank: null,
      },
    }),
    runtimeStatus: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        status: 'Running',
        model: 'test-model',
        dim: 0,
        busy: false,
        error: null,
      },
    }),
    embedJob: vi.fn().mockResolvedValue({ status: 'ok', data: [0, 0] }),
    downloadEmbedder: vi.fn().mockResolvedValue({ status: 'ok', data: null }),
    modelVariants: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        model: 'nomic-embed-text-v1.5',
        runtime: 'ort',
        present: true,
        onnx_present: true,
        mlx_present: null,
      },
    }),
    listEmbedders: vi.fn().mockResolvedValue({
      status: 'ok',
      data: [
        {
          slug: 'nomic-embed-text-v1.5',
          name: 'Nomic Embed',
          memory_mb: 200,
          downloaded: true,
          is_default: true,
          dim: 768,
          max_context_window: 2048,
          source: 'hf:test/test',
          size_bytes: 1,
        },
      ],
    }),
    getJobAd: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        path: '/tmp/jobs/job-1/job-ad.txt',
        text: 'Sample job advertisement.',
        created: false,
      },
    }),
    setJobAdPath: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        path: '/tmp/jobs/job-1/job-ad.txt',
        text: 'Sample job advertisement.',
        created: false,
      },
    }),
    reloadJobAd: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        id: 'job-1',
        title: 'Job 1',
        job_ad_text: 'Sample job advertisement.',
      },
    }),
    watchJobCvDir: vi.fn().mockResolvedValue(undefined),
  },
  unwrapResult: vi.fn((result: { status: string; data?: unknown }) => {
    if (result.status === 'ok') return result.data
    throw result
  }),
}))
