import { Outlet } from '@tanstack/react-router'

import { ActivityBar } from './ActivityBar'
import { StatusBar } from './StatusBar'
import { ModelDownloadModal } from './ModelDownloadModal'
import { CommandPalette } from '@/components/command-palette/CommandPalette'
import { PreferencesDialog } from '@/components/preferences/PreferencesDialog'
import { Toaster } from 'sonner'
import { useTheme } from '@/hooks/use-theme'
import { useMainWindowEventListeners } from '@/hooks/useMainWindowEventListeners'
import { useImportProgressListener } from '@/hooks/useImportProgress'
import { useEmbeddingProgressListener } from '@/hooks/useEmbeddingProgress'
import { useRuntimeStatusListener } from '@/services/runtime'

/**
 * App shell: the routed page filling the middle, status bar
 * at the bottom. No side panels — the Jobs list (home) and the 3-pane workspace
 * are each distinct full-width pages.
 */
export function MainWindow() {
  const { theme } = useTheme()

  useMainWindowEventListeners()
  useImportProgressListener()
  useEmbeddingProgressListener()
  useRuntimeStatusListener()

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden rounded-[var(--app-corner-radius)] bg-background">
      <div className="flex flex-1 flex-row overflow-hidden">
        <ActivityBar />

        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <main className="flex flex-1 flex-col overflow-hidden">
            <Outlet />
          </main>
        </div>
      </div>

      <StatusBar />

      {/* Global UI (hidden until triggered) */}
      <CommandPalette />
      <ModelDownloadModal />
      <PreferencesDialog />
      <Toaster
        position="bottom-right"
        theme={
          theme === 'dark' ? 'dark' : theme === 'light' ? 'light' : 'system'
        }
        className="toaster group"
        toastOptions={{
          classNames: {
            toast:
              'group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg',
            description: 'group-[.toast]:text-muted-foreground',
            actionButton:
              'group-[.toast]:bg-primary group-[.toast]:text-primary-foreground',
            cancelButton:
              'group-[.toast]:bg-muted group-[.toast]:text-muted-foreground',
          },
        }}
      />
    </div>
  )
}
