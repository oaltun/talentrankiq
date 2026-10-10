import { useImportStore } from '@/store/import-store'
import { useRuntimeStatus } from '@/services/runtime'
import { Minimize, Maximize, X, Maximize2 } from 'lucide-react'
import { useCommandContext } from '@/hooks/use-command-context'
import { executeCommand } from '@/lib/commands'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

/** Window control buttons for when native decorations are disabled. */
function WindowControls() {
  const context = useCommandContext()
  const [isMaximized, setIsMaximized] = useState(false)

  useEffect(() => {
    const appWindow = getCurrentWindow()

    appWindow
      .isMaximized()
      .then(setIsMaximized)
      .catch(() => {
        /* ignore */
      })

    let aborted = false
    let resolvedUnsub: (() => void) | null = null

    appWindow
      .onResized(async () => {
        try {
          const maximized = await appWindow.isMaximized()
          if (!aborted) setIsMaximized(maximized)
        } catch {
          // Ignore errors during cleanup
        }
      })
      .then(unsub => {
        if (aborted) {
          unsub()
        } else {
          resolvedUnsub = unsub
        }
      })

    return () => {
      aborted = true
      if (resolvedUnsub) {
        resolvedUnsub()
      }
    }
  }, [])

  const handleClose = async () => {
    await executeCommand('window-close', context)
  }

  const handleMinimize = async () => {
    await executeCommand('window-minimize', context)
  }

  const handleMaximizeToggle = async () => {
    try {
      const appWindow = getCurrentWindow()
      const maximized = await appWindow.isMaximized()
      if (maximized) {
        await appWindow.unmaximize()
        setIsMaximized(false)
      } else {
        await appWindow.maximize()
        setIsMaximized(true)
      }
    } catch {
      await executeCommand('window-toggle-maximize', context)
    }
  }

  const buttonClass =
    'flex h-6 w-6 items-center justify-center transition-colors rounded hover:bg-foreground/10'

  return (
    <div className="flex items-center gap-1 ml-2 border-l pl-2">
      <button
        type="button"
        onClick={handleMinimize}
        className={cn(buttonClass)}
        title="Minimize"
        aria-label="Minimize window"
      >
        <Minimize className="h-3 w-3" />
      </button>
      <button
        type="button"
        onClick={handleMaximizeToggle}
        className={cn(buttonClass)}
        title={isMaximized ? 'Restore' : 'Maximize'}
        aria-label={isMaximized ? 'Restore window' : 'Maximize window'}
      >
        {isMaximized ? (
          <Maximize2 className="h-3 w-3" />
        ) : (
          <Maximize className="h-3 w-3" />
        )}
      </button>
      <button
        type="button"
        onClick={handleClose}
        className={cn(
          buttonClass,
          'hover:bg-destructive hover:text-destructive-foreground'
        )}
        title="Close"
        aria-label="Close window"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  )
}

/** Slim status bar: live backend work + window controls (left), privacy promise (right). */
export function StatusBar() {
  const { t } = useTranslation()
  const progress = useImportStore(s => s.progress)
  const active = Object.entries(progress).filter(([, p]) => p.active)
  const { data: runtime } = useRuntimeStatus()
  // Backend-derived work line: what the engine is doing right now (scout /
  // embed / judge / inspect), with the resolved arm next to the engine chip.
  const activity = runtime?.activity
  const arm =
    runtime?.backend === 'mlx'
      ? t('status.arm.mlx')
      : runtime?.backend === 'ort'
        ? t('status.arm.ort')
        : null

  const engine =
    runtime === undefined
      ? null
      : runtime.status === 'Running'
        ? runtime.busy
          ? {
              label: t('status.engine.working'),
              dot: 'bg-emerald-500 animate-pulse',
              text: 'text-emerald-600',
            }
          : {
              label: t('status.engine.ready'),
              dot: 'bg-emerald-500',
              text: 'text-emerald-600',
            }
        : runtime.status === 'Starting'
          ? {
              label: t('status.engine.loading'),
              dot: 'bg-amber-500 animate-pulse',
              text: 'text-amber-600',
            }
          : runtime.status === 'Stopped'
            ? {
                label: t('status.engine.unloaded'),
                dot: 'bg-muted-foreground/50',
                text: 'text-muted-foreground',
              }
            : {
                label: t('status.engine.error'),
                dot: 'bg-rose-500',
                text: 'text-rose-600',
              }

  const engineTitle =
    runtime?.status === 'Error'
      ? (runtime.error ?? t('status.engine.error'))
      : runtime?.status === 'Stopped'
        ? t('status.engine.unloadedHint')
        : runtime?.status === 'Starting'
          ? t('status.engine.loadingHint')
          : runtime?.status === 'Running'
            ? `${runtime.model ?? ''}${runtime.dim ? ` · ${runtime.dim}` : ''}${arm ? ` · ${arm}` : ''}`
            : undefined

  return (
    <footer className="flex items-center gap-2 border-t bg-muted/30 px-3 py-1 text-xs text-muted-foreground">
      <div className="flex items-center gap-4">
        {engine && (
          <span
            className={cn('flex items-center gap-1.5', engine.text)}
            title={engineTitle}
          >
            <span className={cn('size-2 rounded-full', engine.dot)} />
            {engine.label}
            {arm && (
              <span className="rounded bg-foreground/5 px-1 text-[10px] tracking-wide text-muted-foreground">
                {arm}
              </span>
            )}
          </span>
        )}
        {activity && (
          <span
            className="flex items-center gap-1"
            title={`${activity.stage} · ${activity.subject}${activity.model ? ` · ${activity.model}` : ''}${activity.detail ? ` · ${activity.detail}` : ''}`}
          >
            <span>{t(`status.stage.${activity.stage}`)}</span>
            <span className="max-w-48 truncate text-foreground/80">
              {activity.subject}
            </span>
            {activity.model && (
              <span>
                {t('status.stage.byModel', { model: activity.model })}
              </span>
            )}
            {activity.detail && <span>· {activity.detail}</span>}
          </span>
        )}
        {active.map(([jobId, p]) => (
          <span key={jobId}>
            {p.phase === 'embedding'
              ? t('status.phase.embedding')
              : p.phase === 'ranking'
                ? t('status.phase.ranking')
                : p.phase === 'engine-unavailable'
                  ? t('status.phase.engineUnavailable')
                  : `${t('status.phase.importing')} ${p.done}/${p.total}`}
            {p.failed > 0 ? ` · ${p.failed} ${t('status.phase.failed')}` : ''}
          </span>
        ))}
        <WindowControls />
      </div>
      <span className="ml-auto">All local: no data leaves this machine</span>
    </footer>
  )
}
