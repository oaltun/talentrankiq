import { useState } from 'react'
import {
  Loader2,
  Download,
  Trash2,
  Check,
  Cpu,
  HardDrive,
  FlaskConical,
  Play,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { errorMessage } from '@/lib/errors'
import {
  useEmbedders,
  useDownloadEmbedder,
  useRemoveEmbedder,
  useSetDefaultEmbedder,
  useSetEmbeddingModel,
  useModelDownloadProgress,
  useModelDownloadCompleteToast,
  type GuiEmbedder,
} from '@/services/models'
import { usePreferences } from '@/services/preferences'
import { TryModelDialog } from './TryModelDialog'

function fmtBytes(n: number | null | undefined): string {
  if (!n) return 'unknown'
  const mb = n / (1024 * 1024)
  if (mb < 1024) return `${mb.toFixed(0)} MB`
  return `${(mb / 1024).toFixed(1)} GB`
}

function fmtMemory(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

// Determine arm from slug/backend (not runtime resolve)
function getArmFromSlug(slug: string): string | null {
  if (slug.includes('-MLX-')) return 'MLX'
  if (slug.includes('-ONNX-')) return 'ONNX'
  return null
}

function EmbedderCard({
  embedder,
  isDefault,
  progressPct,
  downloading,
  onDownload,
  onRemove,
  onSetDefault,
  onTry,
  onUseModel,
}: {
  embedder: GuiEmbedder
  isDefault: boolean
  progressPct: number | undefined
  downloading: boolean
  onDownload: (slug: string) => void
  onRemove: (slug: string) => void
  onSetDefault: (slug: string) => void
  onTry: (slug: string, name: string) => void
  onUseModel: (slug: string) => void
}) {
  const arm = getArmFromSlug(embedder.slug)
  const {
    slug,
    name,
    memory_mb,
    downloaded,
    dim,
    max_context_window,
    size_bytes,
  } = embedder
  return (
    <div
      className={cn(
        'flex flex-col gap-2 rounded-md border bg-card p-3 text-sm transition-shadow',
        isDefault && 'ring-2 ring-primary shadow-md'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium">{name}</span>
            {isDefault && (
              <span className="inline-flex items-center gap-1 rounded bg-primary/15 text-xs text-primary">
                <Check className="size-3" /> Default
              </span>
            )}
            {arm && (
              <span className="rounded bg-foreground/5 px-1.5 text-[10px] tracking-wide text-muted-foreground">
                {arm}
              </span>
            )}
          </div>
          <div className="mt-0.5 font-mono text-xs text-muted-foreground break-all">
            {slug}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <HardDrive className="size-3 shrink-0" />
          <span className="flex flex-col">
            <span>{fmtMemory(memory_mb)}</span>
            <span className="text-[10px] text-muted-foreground/70">
              RAM required
            </span>
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Cpu className="size-3 shrink-0" />
          <span className="flex flex-col">
            <span>{dim}d</span>
            <span className="text-[10px] text-muted-foreground/70">
              Embedding dimensions
            </span>
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <span className="flex flex-col">
            <span>
              {max_context_window >= 1000
                ? `${max_context_window / 1000}k`
                : max_context_window}{' '}
              tokens
            </span>
            <span className="text-[10px] text-muted-foreground/70">
              Max context window
            </span>
          </span>
        </div>
        {downloaded && (
          <div className="col-span-2 flex items-center gap-1.5 text-muted-foreground sm:col-span-1">
            <span className="flex flex-col">
              <span>{fmtBytes(size_bytes)}</span>
              <span className="text-[10px] text-muted-foreground/70">
                Disk size
              </span>
            </span>
          </div>
        )}
      </div>

      {progressPct !== undefined && (
        <div className="h-1.5 w-full rounded bg-muted">
          <div
            className="h-full min-w-[12px] rounded bg-primary transition-[width]"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        {downloaded ? (
          <>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onTry(slug, name)}
            >
              <FlaskConical className="size-3" /> Try
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onSetDefault(slug)}
              disabled={isDefault}
            >
              {isDefault ? 'Default set' : 'Set as default'}
            </Button>
            <Button
              size="sm"
              variant="default"
              onClick={() => onUseModel(slug)}
            >
              <Play className="size-3" /> Use this model
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => onRemove(slug)}
            >
              <Trash2 className="size-3" /> Remove
            </Button>
          </>
        ) : (
          <Button
            size="sm"
            onClick={() => onDownload(slug)}
            disabled={downloading}
          >
            {downloading ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Download className="size-3" />
            )}
            Download
          </Button>
        )}
      </div>
    </div>
  )
}

function EmbeddingPane() {
  const { data, isLoading, error } = useEmbedders()
  const { data: prefs } = usePreferences()
  const download = useDownloadEmbedder()
  const remove = useRemoveEmbedder()
  const setDefault = useSetDefaultEmbedder()
  const setEmbeddingModel = useSetEmbeddingModel()
  const progress = useModelDownloadProgress()
  useModelDownloadCompleteToast()

  const defaultSlug: string | null = prefs?.embedding_model ?? null

  const [trySlug, setTrySlug] = useState<string | null>(null)
  const [tryName, setTryName] = useState('')
  const tryEmbedder = trySlug
    ? (data ?? []).find(e => e.slug === trySlug)
    : null

  const handleUseModel = (slug: string) => {
    setEmbeddingModel.mutate(slug)
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading embedders...
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-sm text-destructive">
        Failed to load embedders: {errorMessage(error)}
      </div>
    )
  }

  const embedders = [...(data ?? [])].sort((a, b) => a.memory_mb - b.memory_mb)
  const anyDownloaded = embedders.some(e => e.downloaded)

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        On-device embedding models from the app&apos;s model registry. Download,
        remove, and pick the model used for ranking CVs. The engine starts when
        you click &quot;Use this model&quot; on a downloaded model.
      </p>

      {!anyDownloaded && (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <HardDrive className="size-12 text-muted-foreground/50 mb-4" />
          <p className="text-lg font-medium mb-2">No models downloaded</p>
          <p className="text-sm text-muted-foreground max-w-md">
            Download a model to enable AI ranking. The engine will start when
            you click &quot;Use this model&quot; on a downloaded model.
          </p>
        </div>
      )}

      {embedders.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No embedders configured in triq.yaml.
        </p>
      ) : (
        embedders.map(e => {
          const entries = progress.get(e.slug)
          const p = entries[0]
          const total = p?.total ?? 0
          const pct = total > 0 ? ((p?.done ?? 0) / total) * 100 : undefined
          const downloading = download.isPending && entries.length > 0
          return (
            <EmbedderCard
              key={e.slug}
              embedder={e}
              isDefault={e.slug === defaultSlug}
              progressPct={pct}
              downloading={downloading}
              onDownload={slug => download.mutate(slug)}
              onRemove={slug => remove.mutate(slug)}
              onSetDefault={slug => setDefault.mutate(slug)}
              onTry={(slug, name) => {
                setTrySlug(slug)
                setTryName(name)
              }}
              onUseModel={handleUseModel}
            />
          )
        })
      )}

      <TryModelDialog
        open={!!trySlug}
        onOpenChange={open => {
          if (!open) setTrySlug(null)
        }}
        slug={trySlug ?? ''}
        modelName={tryEmbedder?.name ?? tryName}
      />
    </div>
  )
}

export function ModelsView() {
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-3xl space-y-6">
        <div>
          <h1 className="text-lg font-semibold">Models</h1>
          <p className="text-sm text-muted-foreground">
            Manage the on-device embedding models used for ranking CVs.
          </p>
        </div>

        <EmbeddingPane />
      </div>
    </div>
  )
}
