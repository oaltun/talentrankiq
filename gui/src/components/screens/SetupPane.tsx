import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { errorMessage, isDomainNotFound } from '@/lib/errors'
import type { Criterion, JobDetail } from '@/lib/tauri-bindings'
import {
  useAddCriterion,
  useDeleteCriterion,
  useUpdateCriterion,
} from '@/services/criteria'
import { useSetCvDir } from '@/services/jobs'
import { useImportStore } from '@/store/import-store'
import { useEmbeddingStore } from '@/store/embedding-store'
import { openFileDialog } from '@/lib/file-dialogs'
import { openPath } from '@tauri-apps/plugin-opener'
import { Folder, FolderOpen, SlidersHorizontal, Upload, X } from 'lucide-react'

import { PaneCard } from './PaneCard'
import { AdFileCard } from './AdFileCard'

function CriterionRow({
  jobId,
  criterion,
  index,
}: {
  jobId: string
  criterion: Criterion
  index: number
}) {
  const updateCriterion = useUpdateCriterion(jobId)
  const deleteCriterion = useDeleteCriterion(jobId)
  const [text, setText] = useState(criterion.text)

  const commitText = () => {
    const t = text.trim()
    if (!t) {
      // A criterion without text is meaningless — remove it (don't resurrect
      // the old text), matching the composer's empty-means-discard behavior.
      deleteCriterion.mutate(criterion.id)
      toast('Criterion deleted')
      return
    }
    if (t !== criterion.text) {
      updateCriterion.mutate({
        id: criterion.id,
        text: t,
        weight: criterion.weight,
      })
    }
  }

  return (
    <div className="flex items-start gap-2">
      <span
        className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-md bg-muted text-xs text-muted-foreground"
        title={`Criterion C${index}`}
      >
        C{index}
      </span>
      <Textarea
        value={text}
        onChange={e => setText(e.target.value)}
        onBlur={commitText}
        placeholder="criterion"
        className="h-[76px] min-h-0 flex-1 resize-none field-sizing-fixed"
      />
      <Button
        size="icon-sm"
        variant="ghost"
        onClick={() => {
          deleteCriterion.mutate(criterion.id)
          toast('Criterion deleted')
        }}
      >
        <X />
      </Button>
    </div>
  )
}

/**
 * "Add criterion…" composer. The moment text is typed it grows into a full
 * criterion row (C<n> badge, discard button — identical to CriterionRow) and
 * a fresh empty composer appears below, so the next criterion is ready before
 * this one is committed. Blur (outside the row) persists the criterion;
 * Escape or the ✕ discards the draft. New criteria start at the configured
 * default weight (`rubric_defaults.criterion_points` — weight lives in the
 * Scoring & rubric dialog).
 */
function NewCriterionInput({
  jobId,
  nextIndex,
}: {
  jobId: string
  nextIndex: number
}) {
  const addCriterion = useAddCriterion(jobId)
  const [text, setText] = useState('')
  const wrapRef = useRef<HTMLDivElement>(null)
  const rowRef = useRef<HTMLDivElement>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)
  const started = text.length > 0

  const reset = () => setText('')

  const commit = () => {
    const t = text.trim()
    if (t) addCriterion.mutate({ text: t })
    reset()
  }

  /** Blur-commit unless focus moved within this draft row. */
  const blurCommit = (e: React.FocusEvent) => {
    const rt = e.relatedTarget
    if (rt instanceof Node && rowRef.current?.contains(rt)) return
    const movedToNext = rt instanceof Node && wrapRef.current?.contains(rt)
    commit()
    // Focus moved to the spawned composer below (which this commit removes):
    // put the caret in this field's fresh empty state so typing can continue.
    if (movedToNext) requestAnimationFrame(() => textRef.current?.focus())
  }

  return (
    <div ref={wrapRef} className="contents">
      <div ref={rowRef} className="flex items-start gap-2">
        {started && (
          <span
            className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-md bg-muted text-xs text-muted-foreground"
            title={`Criterion C${nextIndex}`}
          >
            C{nextIndex}
          </span>
        )}
        <Textarea
          ref={textRef}
          value={text}
          onChange={e => setText(e.target.value)}
          onBlur={blurCommit}
          onKeyDown={e => {
            if (e.key === 'Escape') reset()
          }}
          placeholder={started ? 'criterion' : 'Add criterion…'}
          className="h-[76px] min-h-0 flex-1 resize-none field-sizing-fixed"
        />
        {started && (
          <Button
            size="icon-sm"
            variant="ghost"
            // Keep focus in the row so discarding doesn't blur-commit first.
            onMouseDown={e => e.preventDefault()}
            onClick={reset}
            title="Discard criterion"
          >
            <X />
          </Button>
        )}
      </div>
      {started && <NewCriterionInput jobId={jobId} nextIndex={nextIndex + 1} />}
    </div>
  )
}

/** Compact read-only rubric line, e.g. "ad 50 · criteria 10 · FB 10/10".
 * "mixed" when criteria carry differing weights (per-criterion overrides). */
function scoringSummary(detail: JobDetail): string {
  const job = detail.job
  const distinct = [...new Set(detail.criteria.map(c => c.weight))]
  const criteria =
    distinct.length === 0
      ? '—'
      : distinct.length === 1
        ? String(distinct[0])
        : 'mixed'
  return `ad ${job.ad_weight ?? 0} · criteria ${criteria} · FB ${job.fb_job_weight ?? 0}/${job.fb_profile_weight ?? 0}`
}

/// Watched CV folder display: read-only path label + browse + immediate Import.
/// Remounts (via `key` at the usage site) whenever the stored directory changes.
function CvDirectory({
  jobId,
  storedDir,
  importActive,
}: {
  jobId: string
  storedDir: string
  importActive: boolean
}) {
  const setCvDir = useSetCvDir(jobId)

  const commit = (dir: string) => {
    if (dir === storedDir) return
    setCvDir.mutate(dir, {
      onError: e =>
        toast.error(`Could not set CV directory: ${errorMessage(e)}`),
    })
  }

  const handleBrowse = async () => {
    const dir = await openFileDialog({
      directory: true,
      multiple: false,
      startDir: storedDir || undefined,
    })
    if (typeof dir !== 'string') return
    commit(dir)
  }

  /**
   * Re-save the *same* path. The backend re-creates a missing CV directory when
   * it (re)starts the watcher, so this repairs the folder without the user
   * having to browse for it again. Goes through the mutation directly because
   * `commit` is a no-op for an unchanged path.
   */
  const recreate = (dir: string) =>
    setCvDir.mutate(dir, {
      onSuccess: () => toast.success(`Created ${dir}`),
      onError: e =>
        toast.error(`Could not create the folder: ${errorMessage(e)}`),
    })

  const handleImport = async () => {
    const dir = storedDir.trim()
    if (!dir) return
    try {
      unwrapResult(await commands.importCvDirectory(jobId, dir))
    } catch (e) {
      // The folder is app-managed: it was deleted, moved, or lives on an
      // unmounted volume. That's recoverable in one click — offer it instead of
      // a dead-end red error.
      if (isDomainNotFound(e, 'CV directory')) {
        toast.error(errorMessage(e), {
          action: { label: 'Create folder', onClick: () => recreate(dir) },
        })
        return
      }
      toast.error(`Import failed: ${errorMessage(e)}`)
    }
  }

  const handleOpen = async () => {
    const dir = storedDir.trim()
    if (!dir) return
    try {
      await openPath(dir)
    } catch (e) {
      toast.error(`Could not open folder: ${errorMessage(e)}`)
    }
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="min-w-0 cursor-text rounded-md border px-3 py-1 font-mono text-xs break-all text-muted-foreground select-text">
        {storedDir || <span className="italic">No folder selected</span>}
      </div>
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={handleBrowse}>
          <FolderOpen className="size-4" />
          Change
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleOpen}
          disabled={!storedDir.trim()}
          title="Open the folder in your file manager to drop CV files in"
        >
          <Folder className="size-4" />
          Open
        </Button>
        <Button
          size="sm"
          onClick={handleImport}
          disabled={importActive || !storedDir.trim()}
        >
          <Upload className="size-4" />
          Import CVs
        </Button>
      </div>
    </div>
  )
}

export function SetupPane({
  jobId,
  detail,
  onOpenRubric,
}: {
  jobId: string
  detail: JobDetail
  onOpenRubric: () => void
}) {
  const job = detail.job
  const importProgress = useImportStore(s => s.progress[jobId])
  const embedProgress = useEmbeddingStore(s => s.progress[jobId])
  const importPhase = importProgress?.phase
  const importLabel =
    importPhase === 'embedding'
      ? 'Embedding CVs…'
      : importPhase === 'ranking'
        ? 'Ranking candidates…'
        : importPhase === 'engine-unavailable'
          ? 'AI engine unavailable — skipped embedding/ranking'
          : null
  // During the embedding phase the bar + counter follow the per-candidate
  // embed progress (the extraction counter stays pinned at its final value).
  const embedCount =
    importPhase === 'embedding' && embedProgress?.active
      ? `${embedProgress.done}/${embedProgress.total}`
      : null
  const importBarValue =
    importPhase === 'embedding' && embedProgress?.active && embedProgress.total
      ? (embedProgress.done / embedProgress.total) * 100
      : importProgress?.total
        ? (importProgress.done / importProgress.total) * 100
        : 0

  return (
    <section className="flex min-h-[20rem] flex-col gap-3 rounded-lg border bg-muted/40 p-3 md:h-full md:w-[24rem] md:shrink-0 md:overflow-y-auto">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Setup
      </h2>

      {/* The ad is a file the user owns, not a pane: its controls lead the Setup
          pane, with a read-only peek at the contents under them. */}
      <AdFileCard jobId={jobId} />

      <PaneCard title="CV directory">
        <CvDirectory
          key={job.import_directory_path ?? ''}
          jobId={jobId}
          storedDir={job.import_directory_path ?? ''}
          importActive={!!importProgress?.active}
        />
      </PaneCard>

      {importProgress?.active && (
        <PaneCard title="Importing">
          <div className="flex w-full flex-col gap-1">
            <Progress value={importBarValue} />
            <span className="text-center text-xs text-muted-foreground">
              {importLabel ??
                `${importProgress.done} / ${importProgress.total} done`}
              {embedCount != null ? ` · ${embedCount}` : ''}
              {importProgress.failed
                ? ` · ${importProgress.failed} failed`
                : ''}
              {importProgress.phase === 'extracting' ? ' · extracting' : ''}
            </span>
          </div>
        </PaneCard>
      )}

      <PaneCard title="Additional criteria">
        <div className="flex flex-col gap-1.5">
          {detail.criteria.map((c, i) => (
            <CriterionRow
              key={c.id}
              jobId={jobId}
              criterion={c}
              index={i + 1}
            />
          ))}
          <NewCriterionInput
            key={jobId}
            jobId={jobId}
            nextIndex={detail.criteria.length + 1}
          />
        </div>
      </PaneCard>

      <PaneCard title="Scoring & rubric">
        <div className="flex items-center justify-between gap-2">
          <span
            className="min-w-0 truncate font-mono text-xs text-muted-foreground"
            title={`Rubric weights: ${scoringSummary(detail)}`}
          >
            {scoringSummary(detail)}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="shrink-0"
            onClick={onOpenRubric}
          >
            <SlidersHorizontal className="size-3.5" />
            Edit
          </Button>
        </div>
      </PaneCard>
    </section>
  )
}
