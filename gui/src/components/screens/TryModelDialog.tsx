import { useState, useEffect } from 'react'
import { Loader2, FlaskConical } from 'lucide-react'
import { listen } from '@tauri-apps/api/event'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { commands } from '@/lib/tauri-bindings'
import type { CosinePair, TryModelResult } from '@/lib/bindings'
import { cn } from '@/lib/utils'

interface TryModelProgressEvent {
  cvIndex: number
  cvTotal: number
  cvLabel: string
}

function SimilarityMatrix({ result }: { result: TryModelResult }) {
  const { labels, pairs } = result
  const n = labels.length
  const matrix: (number | null)[][] = Array.from({ length: n }, () =>
    Array.from({ length: n }, () => null)
  )
  for (const p of pairs) {
    if (p.i < n && p.j < n) {
      const rowI = matrix[p.i]
      const rowJ = matrix[p.j]
      if (rowI) rowI[p.j] = p.similarity
      if (rowJ) rowJ[p.i] = p.similarity
    }
  }

  function cellColor(val: number | null): string {
    if (val === null) return 'text-muted-foreground/30'
    if (val > 0.95) return 'bg-green-500/20 text-green-700 dark:text-green-400'
    if (val > 0.8)
      return 'bg-yellow-500/15 text-yellow-700 dark:text-yellow-400'
    if (val > 0.6)
      return 'bg-orange-500/10 text-orange-700 dark:text-orange-400'
    return 'text-muted-foreground'
  }

  const shortLabels = labels.map(l =>
    l.replace(/^(CV \d+ — )/, '$1').slice(0, 18)
  )

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1 text-left font-normal text-muted-foreground sticky left-0 bg-background z-10"></th>
            {shortLabels.map((l, i) => (
              <th
                key={i}
                className="p-1 text-left font-normal text-muted-foreground whitespace-nowrap"
                title={labels[i]}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td
                className="p-1 font-normal text-muted-foreground whitespace-nowrap sticky left-0 bg-background z-10"
                title={labels[i]}
              >
                {shortLabels[i]}
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  className={cn(
                    'p-1 font-mono whitespace-nowrap text-center min-w-[4.5rem]',
                    cellColor(val)
                  )}
                  title={
                    val !== null ? `${labels[i]} × ${labels[j]}` : undefined
                  }
                >
                  {i === j ? '1.000' : (val?.toFixed(3) ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PairList({ pairs }: { pairs: CosinePair[] }) {
  const sorted = [...pairs].sort((a, b) => b.similarity - a.similarity)
  return (
    <div className="space-y-1 text-xs max-h-60 overflow-y-auto">
      {sorted.map((p, idx) => (
        <div
          key={idx}
          className={cn(
            'flex items-center gap-2 px-2 py-1 rounded',
            p.similarity > 0.99
              ? 'bg-green-500/15 text-green-700 dark:text-green-400'
              : 'text-muted-foreground'
          )}
        >
          <span className="font-mono w-14 text-right shrink-0">
            {p.similarity.toFixed(4)}
          </span>
          <span className="truncate" title={`${p.label_i} × ${p.label_j}`}>
            {p.label_i} ↔ {p.label_j}
          </span>
        </div>
      ))}
    </div>
  )
}

export function TryModelDialog({
  open,
  onOpenChange,
  slug,
  modelName,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  slug: string
  modelName: string
}) {
  const [result, setResult] = useState<TryModelResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<TryModelProgressEvent | null>(null)

  useEffect(() => {
    if (!loading) return
    let unlisten: (() => void) | null = null
    listen<TryModelProgressEvent>('try-model-progress', e => {
      setProgress(e.payload)
    })
      .then(fn => {
        unlisten = fn
      })
      .catch(() => undefined)
    return () => {
      unlisten?.()
    }
  }, [loading])

  async function run() {
    setLoading(true)
    setError(null)
    setResult(null)
    setProgress(null)
    const res = await commands.tryModel(slug)
    if (res.status === 'ok') {
      setResult(res.data)
    } else {
      setError(res.error)
    }
    setProgress(null)
    setLoading(false)
  }

  function handleClose(nextOpen: boolean) {
    if (!nextOpen) {
      setResult(null)
      setError(null)
      setProgress(null)
    }
    onOpenChange(nextOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="w-[95vw] max-w-[95vw] sm:!max-w-[95vw] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            <span className="inline-flex items-center gap-2">
              <FlaskConical className="size-4" />
              Try {modelName}
            </span>
          </DialogTitle>
          <DialogDescription>
            Embeds {11} real anonymized CVs and computes pair-wise cosine
            similarities. Two CVs differ by a single word.
          </DialogDescription>
        </DialogHeader>

        {!result && !loading && !error && (
          <div className="py-8 text-center">
            <p className="text-sm text-muted-foreground mb-4">
              This will embed {11} large CVs using{' '}
              <span className="font-mono">{slug}</span> and show cosine
              similarities between them.
            </p>
            <Button onClick={run}>
              <FlaskConical className="size-4" />
              Run benchmark
            </Button>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center gap-2 py-8 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            {progress ? (
              <span>
                Embedding CV {progress.cvIndex + 1}/{progress.cvTotal}…
              </span>
            ) : (
              <span>Embedding CVs…</span>
            )}
          </div>
        )}

        {error && <div className="py-4 text-sm text-destructive">{error}</div>}

        {result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="rounded-md border p-3">
                <div className="text-muted-foreground text-xs">
                  Median time per CV
                </div>
                <div className="text-lg font-mono font-medium mt-1">
                  {(result.median_ms / 1000).toFixed(2)} s
                </div>
              </div>
              <div className="rounded-md border p-3">
                <div className="text-muted-foreground text-xs">
                  Per-CV times
                </div>
                <div className="font-mono text-xs mt-1 leading-relaxed">
                  {result.per_cv_ms.map((ms, i) => (
                    <span key={i}>
                      {(ms / 1000).toFixed(2)}
                      {i < result.per_cv_ms.length - 1 ? ' · ' : ''}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium mb-2">
                Cosine similarity matrix
              </h3>
              <SimilarityMatrix result={result} />
            </div>

            <div>
              <h3 className="text-sm font-medium mb-2">
                All pairs (sorted by similarity)
              </h3>
              <PairList pairs={result.pairs} />
            </div>

            <div className="flex justify-center pt-2">
              <Button variant="outline" onClick={run}>
                <FlaskConical className="size-3" />
                Run again
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
