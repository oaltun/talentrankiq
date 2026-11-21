import { useState, useEffect, useRef } from "react"
import { Download, Check, Loader2, Trash2, HardDrive } from "lucide-react"
import {
  getScoutModel,
  downloadEmbedder,
  removeEmbedder,
  listenModelDownloadProgress,
  listenModelDownloadComplete,
  type ScoutModel,
} from "../../../services/models"
import { Button } from "../../ui/button"
import { useI18n } from "../../../i18n/i18n-context"

function fmtBytes(n: number | null | undefined, unknownLabel: string): string {
  if (n === null || n === undefined || n === 0) return unknownLabel
  const mb = n / (1024 * 1024)
  if (mb < 1024) return `${mb.toFixed(0)} MB`
  return `${(mb / 1024).toFixed(1)} GB`
}

export function ScoutModelCard() {
  const { t } = useI18n()
  const [quick, setScout] = useState<ScoutModel | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [downloadStatus, setDownloadStatus] = useState<{
    done_bytes: number
    total_bytes: number | null
    filename: string
    is_downloading: boolean
  } | null>(null)
  const downloadStartRef = useRef(0)
  const lastPctRef = useRef(-1)

  const load = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const data = await getScoutModel()
      setScout(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : t("models.quick.loadError"))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  useEffect(() => {
    if (!quick) return
    let unlistenProgressFn: (() => void) | null = null
    let unlistenCompleteFn: (() => void) | null = null

    listenModelDownloadProgress((event) => {
      if (event.repo_id === quick.slug) {
        const pct =
          event.total && event.total > 0
            ? Math.floor((event.done / event.total) * 100)
            : 0
        if (
          pct === 0 ||
          (pct === 100 && lastPctRef.current < 100) ||
          (pct < 100 && pct - lastPctRef.current >= 10)
        ) {
          lastPctRef.current = pct
          console.log(
            `[quick-model] download ${pct}% (${event.done}/${event.total ?? "?"})`
          )
        }
        setDownloadStatus({
          done_bytes: event.done,
          total_bytes: event.total,
          filename: event.filename,
          is_downloading:
            event.total === null || event.done < (event.total ?? 0),
        })
      }
    }).then((fn) => {
      unlistenProgressFn = fn
    })

    listenModelDownloadComplete((event) => {
      if (event.slug === quick.slug) {
        if (downloadStartRef.current) {
          console.log(
            `[quick-model] download complete ${event.slug} in ${Math.round(performance.now() - downloadStartRef.current)}ms`
          )
          downloadStartRef.current = 0
        }
        setScout((prev) => (prev ? { ...prev, downloaded: true } : prev))
        setTimeout(() => setDownloadStatus(null), 2000)
      }
    }).then((fn) => {
      unlistenCompleteFn = fn
    })

    return () => {
      unlistenProgressFn?.()
      unlistenCompleteFn?.()
    }
  }, [quick?.slug])

  const handleDownload = async () => {
    if (!quick) return
    downloadStartRef.current = performance.now()
    lastPctRef.current = -1
    console.log(`[quick-model] download start ${quick.slug}`)
    setDownloadStatus({
      done_bytes: 0,
      total_bytes: null,
      filename: "",
      is_downloading: true,
    })
    try {
      await downloadEmbedder(quick.slug)
      console.log(
        `[quick-model] downloadEmbedder returned after ${Math.round(performance.now() - downloadStartRef.current)}ms`
      )
      await load()
    } catch (e) {
      console.log(
        `[quick-model] download failed: ${e instanceof Error ? e.message : e}`
      )
      setDownloadStatus(null)
      setError(
        e instanceof Error ? e.message : t("models.quick.downloadFailed")
      )
    }
  }

  const handleRemove = async () => {
    if (!quick) return
    try {
      await removeEmbedder(quick.slug)
      setScout((prev) =>
        prev ? { ...prev, downloaded: false, size_bytes: 0 } : prev
      )
      setDownloadStatus(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : t("models.quick.removeFailed"))
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        {t("models.quick.loading")}
      </div>
    )
  }

  if (error) {
    return <div className="text-sm text-destructive">{error}</div>
  }

  if (!quick) {
    return (
      <p className="text-sm text-muted-foreground">
        {t("models.quick.disabled")}
      </p>
    )
  }

  const isDownloading = downloadStatus?.is_downloading === true
  const doneBytes = downloadStatus?.done_bytes ?? 0
  const totalBytes = downloadStatus?.total_bytes ?? null
  const progressPercent =
    totalBytes && totalBytes > 0 ? (doneBytes / totalBytes) * 100 : 0
  const progressText = isDownloading
    ? `${fmtBytes(doneBytes, t("models.size_unknown"))} / ${fmtBytes(totalBytes, t("models.size_unknown"))}`
    : undefined
  const filename = downloadStatus?.filename || ""

  return (
    <div className="flex flex-col gap-2 rounded-md border bg-card p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <span className="font-medium">{quick.name}</span>
          <div className="mt-0.5 font-mono text-xs break-all text-muted-foreground">
            {quick.slug}
          </div>
        </div>
        {quick.downloaded && (
          <span className="inline-flex items-center gap-1 rounded bg-primary/15 text-xs text-primary">
            <Check className="size-3" /> {t("models.downloaded")}
          </span>
        )}
      </div>
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <HardDrive className="size-3 shrink-0" />
        <span>{fmtBytes(quick.size_bytes, t("models.size_unknown"))}</span>
      </div>

      {isDownloading && (
        <div className="flex flex-col gap-1">
          {filename && (
            <span className="truncate text-xs text-muted-foreground">
              {filename}
            </span>
          )}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            <span>{progressText}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        {quick.downloaded ? (
          <>
            <span className="text-xs text-muted-foreground">
              {t("models.ready")}
            </span>
            <Button
              data-testid={`model-remove-${quick.slug}`}
              onClick={handleRemove}
              variant="destructive"
              size="sm"
            >
              <Trash2 className="size-3" /> {t("models.remove")}
            </Button>
          </>
        ) : (
          <Button
            data-testid={`model-download-${quick.slug}`}
            onClick={handleDownload}
            disabled={isDownloading}
            variant="default"
            size="sm"
          >
            <Download className="size-3" />
            {isDownloading ? t("models.downloading") : t("models.download")}
          </Button>
        )}
      </div>
    </div>
  )
}
