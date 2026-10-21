import { useTranslation } from 'react-i18next'
import { AlertTriangle, Download, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Progress } from '@/components/ui/progress'
import {
  useModelDownloadProgress,
  useStartupModelDownload,
} from '@/services/models'

/**
 * Startup model dialog. When the default embedding model's effective arm is
 * missing from the model cache, the dialog asks for consent FIRST — nothing
 * downloads and the engine stays unloaded until the user agrees. Consent
 * starts the backend download and this dialog shows its progress, closing
 * when the model is on disk (or on retry after a failure). Declining keeps
 * the app usable but unrankable (engine not ready) — it never downloads in
 * the background.
 */
export function ModelDownloadModal() {
  const { t } = useTranslation()
  const { status, model, consent, error, retry, accept, decline } =
    useStartupModelDownload()
  const progress = useModelDownloadProgress()
  const entries = model ? progress.get(model.slug) : []
  const name = model?.name ?? model?.slug ?? ''
  const armLabel =
    consent?.variants.runtime === 'mlx'
      ? t('status.arm.mlx')
      : t('status.arm.ort')

  // Compute combined progress across all files for this model
  let combinedPct: number | undefined
  let currentFile: string | undefined
  if (entries.length > 0) {
    let totalDone = 0
    let totalSize = 0
    let hasTotal = true
    for (const entry of entries) {
      totalDone += entry.done
      if (entry.total !== null && entry.total > 0) {
        totalSize += entry.total
      } else {
        hasTotal = false
      }
    }
    if (hasTotal && totalSize > 0) {
      combinedPct = Math.min(100, Math.round((totalDone / totalSize) * 100))
    }
    // Show the currently downloading file
    const downloading = entries.find(e => e.total === null || e.done < (e.total ?? 0))
    currentFile = downloading?.filename
  }

  return (
    <Dialog
      open={
        status === 'consent' || status === 'downloading' || status === 'error'
      }
    >
      <DialogContent
        className="sm:max-w-md"
        showCloseButton={false}
        onInteractOutside={e => e.preventDefault()}
        onEscapeKeyDown={e => e.preventDefault()}
      >
        {status === 'consent' ? (
          <>
            <DialogHeader>
              <DialogTitle>{t('modelDownload.consentTitle', { name })}</DialogTitle>
              <DialogDescription>
                {t('modelDownload.consentDescription', {
                  name,
                  arm: armLabel,
                })}
              </DialogDescription>
            </DialogHeader>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={decline}>
                {t('modelDownload.decline')}
              </Button>
              <Button onClick={() => void accept()}>
                <Download className="size-4" />
                {t('modelDownload.accept')}
              </Button>
            </div>
          </>
        ) : status === 'downloading' ? (
          <>
            <DialogHeader>
              <DialogTitle>{t('modelDownload.title')}</DialogTitle>
              <DialogDescription>
                {t('modelDownload.description', { name })}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="size-4 animate-spin" />
                {currentFile ? (
                  <>
                    {t('modelDownload.downloadingFile', { file: currentFile })}
                  </>
                ) : (
                  t('modelDownload.downloading')
                )}
              </div>
              {combinedPct === undefined ? (
                <div className="h-2 w-full overflow-hidden rounded-full bg-primary/20">
                  <div className="h-full w-1/3 animate-pulse rounded-full bg-primary" />
                </div>
              ) : (
                <Progress value={combinedPct} />
              )}
              <p className="text-xs text-muted-foreground">
                {t('modelDownload.hint')}
              </p>
            </div>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{t('modelDownload.title')}</DialogTitle>
              <DialogDescription>
                {t('modelDownload.description', { name })}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <div className="flex items-start gap-2 text-sm text-destructive">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <span>{error ?? t('modelDownload.failed')}</span>
              </div>
              <Button onClick={retry}>{t('modelDownload.retry')}</Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
