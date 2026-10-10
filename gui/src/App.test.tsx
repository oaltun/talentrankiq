import { render, screen } from '@/test/test-utils'
import { describe, it, expect } from 'vitest'
import App from './App'

// Tauri bindings are mocked globally in src/test/setup.ts

describe('App', () => {
  it('renders the jobs list (index route)', async () => {
    render(<App />)
    // The Jobs screen heading appears once the list query settles.
    expect(
      await screen.findByRole('heading', { name: /^Jobs$/ })
    ).toBeInTheDocument()
  })

  it('renders title bar with traffic light buttons', async () => {
    render(<App />)
    // Wait for splash screen to disappear and app to render
    await screen.findByRole('heading', { name: /^Jobs$/ })
    // Find specifically the window control buttons in the title bar
    const titleBarButtons = screen
      .getAllByRole('button')
      .filter(
        button =>
          button.getAttribute('aria-label')?.includes('window') ||
          button.className.includes('window-control')
      )
    // Should have at least the window control buttons
    expect(titleBarButtons.length).toBeGreaterThan(0)
  })
})
