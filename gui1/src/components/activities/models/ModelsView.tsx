import { useState, useEffect, useRef } from "react"
import {
  Download,
  Check,
  Loader2,
  Trash2,
  HardDrive,
  Cpu,
  Gavel,
} from "lucide-react"
import { open } from "@tauri-apps/plugin-dialog"
import {
  listEmbedders,
  downloadEmbedder,
  removeEmbedder,
  makeDefaultEmbedder,
  loadComparatorArtifact,
  removeComparatorArtifact,
  listenModelDownloadProgress,
  listenModelDownloadComplete,
  type GuiEmbedder,
} from "../../../services/models"
import { errMsg } from "../../../lib/err-msg"
import { Button } from "../../ui/button"
import { useI18n } from "../../../i18n/i18n-context"

function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || n === 0) return "unknown"
  const mb = n / (1024 * 1024)
  if (mb < 1024) return `${mb.toFixed(0)} MB`
  return `${(mb / 1024).toFixed(1)} GB`
}

function fmtMemory(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

interface EmbedderCardProps {
  embedder: GuiEmbedder
  isDefault: boolean
  downloadStatus: {
    done_bytes: number
    total_bytes: number | null
    filename: string
    is_downloading: boolean
  } | null
  comparatorBusy: boolean
  comparatorError: string | null
  onDownload: (slug: string) => void
  onRemove: (slug: string) => void
  onMakeDefault: (slug: string) => void
  onAddComparator: (slug: string) => void
  onRemoveComparator: (slug: string) => void
}

function EmbedderCard({
  embedder,
  isDefault,
  downloadStatus,
  comparatorBusy,
  comparatorError,
  onDownload,
  onRemove,
  onMakeDefault,
  onAddComparator,
  onRemoveComparator,
}: EmbedderCardProps) {
  const { t } = useI18n()
  const {
    slug,
    name,
    memory_mb,
    downloaded,
    dim,
    max_context_window,
    size_bytes,
    comparator,
  } = embedder

  const isDownloading = downloadStatus?.is_downloading === true
  const doneBytes = downloadStatus?.done_bytes ?? 0
  const totalBytes = downloadStatus?.total_bytes ?? null
  const progressPercent =
    totalBytes && totalBytes > 0 ? (doneBytes / totalBytes) * 100 : 0
  const progressText = isDownloading
    ? `${fmtBytes(doneBytes)} / ${fmtBytes(totalBytes)}`
    : undefined
  const filename = downloadStatus?.filename || ""
  const expectedComparatorName = comparator?.path
    ? (comparator.path.split(/[\\/]/).pop() ?? comparator.path)
    : null

  return (
    <div
      className={`flex flex-col gap-2 rounded-md border bg-card p-3 text-sm transition-shadow ${isDefault ? "shadow-md ring-2 ring-primary" : ""}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium">{name}</span>
            {isDefault && (
              <span
                data-testid={`model-default-badge-${slug}`}
                className="inline-flex items-center gap-1 rounded bg-primary/15 text-xs text-primary"
              >
                <Check className="size-3" /> {t("models.default")}
              </span>
            )}
          </div>
          <div className="mt-0.5 font-mono text-xs break-all text-muted-foreground">
            {slug}
          </div>
        </div>
      </div>

      <div
        className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3"
        data-testid={`specs-grid-${slug}`}
      >
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <HardDrive className="size-3 shrink-0" />
          <span className="flex flex-col">
            <span>{fmtMemory(memory_mb)}</span>
            <span className="text-[10px] text-muted-foreground/70">
              {t("models.ram_required")}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Cpu className="size-3 shrink-0" />
          <span className="flex flex-col">
            <span>{dim}d</span>
            <span className="text-[10px] text-muted-foreground/70">
              {t("models.embedding_dimensions")}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <span className="flex flex-col">
            <span>
              {max_context_window >= 1000
                ? `${max_context_window / 1000}k`
                : max_context_window}{" "}
              tokens
            </span>
            <span className="text-[10px] text-muted-foreground/70">
              {t("models.max_context_window")}
            </span>
          </span>
        </div>
        {downloaded && (
          <div className="col-span-2 flex items-center gap-1.5 text-muted-foreground sm:col-span-1">
            <span className="flex flex-col">
              <span>{fmtBytes(size_bytes)}</span>
              <span className="text-[10px] text-muted-foreground/70">
                {t("models.disk_size")}
              </span>
            </span>
          </div>
        )}
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

      <div
        className="flex items-center gap-2 pt-1"
        data-testid={`card-actions-${slug}`}
      >
        {!isDefault && (
          <Button
            data-testid={`model-set-default-${slug}`}
            onClick={() => onMakeDefault(slug)}
            variant="outline"
            size="sm"
          >
            {t("models.set_default")}
          </Button>
        )}
        {downloaded && (
          <Button
            data-testid={`model-remove-${slug}`}
            onClick={() => onRemove(slug)}
            variant="destructive"
            size="sm"
          >
            <Trash2 className="size-3" /> {t("models.remove")}
          </Button>
        )}
        {!downloaded && (
          <Button
            data-testid={`model-download-${slug}`}
            onClick={() => onDownload(slug)}
            disabled={isDownloading}
            variant="default"
            size="sm"
          >
            <Download className="size-3" />
            {t("models.download")}
          </Button>
        )}
      </div>

      {comparator && (
        <div
          className={`flex flex-col gap-1.5 rounded-md border p-2 ${comparator.exists ? "bg-muted/30" : "border-primary/40 bg-primary/5"}`}
          data-testid={`comparator-row-${slug}`}
        >
          <div
            className="flex items-center gap-1.5 text-[10px] tracking-wide text-muted-foreground uppercase"
            data-testid={`comparator-section-label-${slug}`}
          >
            <Gavel className="size-3 shrink-0" />
            {t("models.comparator.section_label")}
          </div>
          {!comparator.exists && (
            <>
              <div className="flex items-center gap-1.5">
                <span
                  data-testid={`comparator-cta-badge-${slug}`}
                  className="inline-flex items-center gap-1 rounded bg-primary/15 px-1.5 py-0.5 text-xs text-primary"
                >
                  <Gavel className="size-3 shrink-0" />
                  {t("models.comparator.pro")}
                </span>
                <span
                  data-testid={`comparator-cta-title-${slug}`}
                  className="text-xs font-medium"
                >
                  {t("models.comparator.unlock")}
                </span>
              </div>
              <span
                data-testid={`comparator-cta-sub-${slug}`}
                className="text-[10px] text-muted-foreground"
              >
                {t("models.comparator.sharper_than")}
              </span>
            </>
          )}
          <div className="flex min-w-0 items-center gap-2">
            <span className="inline-flex shrink-0 items-center gap-1 rounded bg-secondary px-1.5 py-0.5 text-xs text-muted-foreground">
              <Gavel className="size-3 shrink-0" />
              Comparator:{" "}
              {comparator.exists
                ? t("models.comparator.present")
                : t("models.comparator.missing")}
            </span>
            <span className="truncate font-mono text-[10px] text-muted-foreground/70">
              {comparator.path}
            </span>
          </div>
          {expectedComparatorName && (
            <span
              data-testid={`comparator-expected-${slug}`}
              className="text-[10px] text-muted-foreground"
            >
              {t("models.comparator.file_must_be_named")}{" "}
              <span className="font-mono">{expectedComparatorName}</span>
            </span>
          )}
          <div className="flex items-center gap-2">
            {!comparator.exists && (
              <Button
                data-testid={`comparator-add-${slug}`}
                onClick={() => onAddComparator(slug)}
                disabled={comparatorBusy}
                title={
                  expectedComparatorName
                    ? `Pick the comparator artifact file named ${expectedComparatorName}`
                    : undefined
                }
                variant="default"
                size="sm"
              >
                {comparatorBusy
                  ? t("models.comparator.loading")
                  : t("models.comparator.get")}
              </Button>
            )}
            {comparator.exists && (
              <Button
                data-testid={`comparator-remove-${slug}`}
                onClick={() => onRemoveComparator(slug)}
                disabled={comparatorBusy}
                variant="destructive"
                size="sm"
              >
                <Trash2 className="size-3" /> {t("models.remove")}
              </Button>
            )}
          </div>
          {comparatorError && (
            <div
              data-testid={`comparator-error-${slug}`}
              role="alert"
              className="rounded border border-destructive bg-destructive/10 px-2 py-1 text-xs break-all text-destructive"
            >
              {comparatorError}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function ModelsView() {
  const { t } = useI18n()

  const [embedders, setEmbedders] = useState<GuiEmbedder[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [downloadStatuses, setDownloadStatuses] = useState<
    Record<
      string,
      {
        done_bytes: number
        total_bytes: number | null
        filename: string
        is_downloading: boolean
      }
    >
  >({})
  const [defaultSlug, setDefaultSlug] = useState<string | null>(null)
  const [comparatorBusySlugs, setComparatorBusySlugs] = useState<
    Record<string, boolean>
  >({})
  const [comparatorErrors, setComparatorErrors] = useState<
    Record<string, string>
  >({})
  const lastDownloadLogRef = useRef<Record<string, number>>({})
  const downloadStartRef = useRef<Record<string, number>>({})
  const completeLoggedRef = useRef<Set<string>>(new Set())

  const loadEmbedders = async () => {
    try {
      setIsLoading(true)
      const data = await listEmbedders()
      setEmbedders(data)
      const defaultEmbedder = data.find((e) => e.is_default)
      if (defaultEmbedder) {
        setDefaultSlug(defaultEmbedder.slug)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load embedders")
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadEmbedders()
  }, [])

  useEffect(() => {
    let unlistenProgressFn: (() => void) | null = null
    let unlistenCompleteFn: (() => void) | null = null

    listenModelDownloadProgress((event) => {
      const pct =
        event.total && event.total > 0
          ? Math.floor((event.done / event.total) * 100)
          : 0
      const last = lastDownloadLogRef.current[event.repo_id] ?? -1
      if (
        pct === 0 ||
        (pct === 100 && last < 100) ||
        (pct < 100 && pct - last >= 10)
      ) {
        lastDownloadLogRef.current[event.repo_id] = pct
        console.log(
          `[models] download ${event.repo_id} ${pct}% (${event.done}/${event.total ?? "?"})`
        )
      }
      setDownloadStatuses((prev) => ({
        ...prev,
        [event.repo_id]: {
          done_bytes: event.done,
          total_bytes: event.total,
          filename: event.filename,
          is_downloading:
            event.total === null || event.done < (event.total ?? 0),
        },
      }))
    }).then((fn) => {
      unlistenProgressFn = fn
    })

    listenModelDownloadComplete((event) => {
      if (!completeLoggedRef.current.has(event.slug)) {
        completeLoggedRef.current.add(event.slug)
        const started = downloadStartRef.current[event.slug]
        console.log(
          `[models] download complete ${event.slug} in ${started ? Math.round(performance.now() - started) : "?"}ms`
        )
      }
      setEmbedders((prev) =>
        prev.map((e) =>
          e.slug === event.slug ? { ...e, downloaded: true, size_bytes: 1 } : e
        )
      )
      setTimeout(() => {
        setDownloadStatuses((prev) => {
          const next = { ...prev }
          delete next[event.slug]
          return next
        })
      }, 2000)
    }).then((fn) => {
      unlistenCompleteFn = fn
    })

    return () => {
      unlistenProgressFn?.()
      unlistenCompleteFn?.()
    }
  }, [])

  const handleDownload = async (slug: string) => {
    downloadStartRef.current[slug] = performance.now()
    lastDownloadLogRef.current[slug] = -1
    completeLoggedRef.current.delete(slug)
    console.log(`[models] download start ${slug}`)
    setDownloadStatuses((prev) => ({
      ...prev,
      [slug]: {
        done_bytes: 0,
        total_bytes: null,
        filename: "",
        is_downloading: true,
      },
    }))
    try {
      await downloadEmbedder(slug)
      console.log(
        `[models] downloadEmbedder returned for ${slug} after ${Math.round(performance.now() - downloadStartRef.current[slug])}ms`
      )
      await loadEmbedders()
    } catch (e) {
      console.log(
        `[models] download failed for ${slug}: ${e instanceof Error ? e.message : e}`
      )
      setDownloadStatuses((prev) => {
        const next = { ...prev }
        delete next[slug]
        return next
      })
      setError(e instanceof Error ? e.message : "Download failed")
    }
  }

  const handleRemove = async (slug: string) => {
    try {
      await removeEmbedder(slug)
      setEmbedders((prev) =>
        prev.map((e) =>
          e.slug === slug ? { ...e, downloaded: false, size_bytes: 0 } : e
        )
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : "Remove failed")
    }
  }

  const handleMakeDefault = async (slug: string) => {
    try {
      await makeDefaultEmbedder(slug)
      setDefaultSlug(slug)
      setEmbedders((prev) =>
        prev.map((e) => ({ ...e, is_default: e.slug === slug }))
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to set default")
    }
  }

  const handleAddComparator = async (slug: string) => {
    try {
      const selected = await open({
        filters: [{ name: "JSON", extensions: ["json"] }],
      })
      if (!selected || Array.isArray(selected)) return
      setComparatorBusySlugs((prev) => ({ ...prev, [slug]: true }))
      setComparatorErrors((prev) => {
        const next = { ...prev }
        delete next[slug]
        return next
      })
      await loadComparatorArtifact(selected as string, slug)
      await loadEmbedders()
    } catch (e) {
      // Per-card error: the card list stays visible so the user can retry.
      setComparatorErrors((prev) => ({
        ...prev,
        [slug]: errMsg(e, "Failed to load comparator artifact"),
      }))
    } finally {
      setComparatorBusySlugs((prev) => ({ ...prev, [slug]: false }))
    }
  }

  const handleRemoveComparator = async (slug: string) => {
    try {
      setComparatorBusySlugs((prev) => ({ ...prev, [slug]: true }))
      setComparatorErrors((prev) => {
        const next = { ...prev }
        delete next[slug]
        return next
      })
      await removeComparatorArtifact(slug)
      await loadEmbedders()
    } catch (e) {
      setComparatorErrors((prev) => ({
        ...prev,
        [slug]: errMsg(e, "Failed to remove comparator artifact"),
      }))
    } finally {
      setComparatorBusySlugs((prev) => ({ ...prev, [slug]: false }))
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        {t("models.loadingEmbedders")}
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-sm text-destructive">
        {t("models.error.loadEmbedders")}: {error}
      </div>
    )
  }

  const sortedEmbedders = [...embedders].sort(
    (a, b) => a.memory_mb - b.memory_mb
  )
  const anyDownloaded = sortedEmbedders.some((e) => e.downloaded)

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">{t("models.description")}</p>

      {!anyDownloaded && (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <HardDrive className="mb-4 size-12 text-muted-foreground/50" />
          <p className="mb-2 text-lg font-medium">
            {t("models.noModelsDownloaded")}
          </p>
          <p className="max-w-md text-sm text-muted-foreground">
            {t("models.noModelsDescription")}
          </p>
        </div>
      )}

      {sortedEmbedders.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {t("models.noEmbeddersConfigured")}
        </p>
      ) : (
        sortedEmbedders.map((e) => (
          <EmbedderCard
            key={e.slug}
            embedder={e}
            isDefault={e.slug === defaultSlug}
            downloadStatus={downloadStatuses[e.slug] || null}
            comparatorBusy={comparatorBusySlugs[e.slug] === true}
            comparatorError={comparatorErrors[e.slug] ?? null}
            onDownload={handleDownload}
            onRemove={handleRemove}
            onMakeDefault={handleMakeDefault}
            onAddComparator={handleAddComparator}
            onRemoveComparator={handleRemoveComparator}
          />
        ))
      )}
    </div>
  )
}
