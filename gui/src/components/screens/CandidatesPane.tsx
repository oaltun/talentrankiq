import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { openPath } from '@tauri-apps/plugin-opener'
import { saveFileDialog } from '@/lib/file-dialogs'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { errorMessage } from '@/lib/errors'
import { fmtScore, fmtWeight, per100 } from '@/lib/scores'
import type {
  AdSource,
  GradeSpace,
  JobDetail,
  RubricQuestion,
} from '@/lib/tauri-bindings'
import { useJobAdFile, useUpdateJob, useJobUiState } from '@/services/jobs'
import { useExportResultsCsv } from '@/services/results-csv'
import { useRankJob } from '@/services/rank'
import { useCandidatesPage, useEmptyCandidates } from '@/services/candidates'
import { useLikedCandidateIds, useLikeCandidate } from '@/services/feedback'
import { useDebouncedValue } from '@/hooks/use-debounced-value'
import { Paginator } from '@/components/ui/paginator'
import { selectLikes, useFeedbackStore } from '@/store/feedback-store'
import { useImportStore } from '@/store/import-store'
import { useUIStore } from '@/store/ui-store'
import {
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronDown,
  Download,
  Eraser,
  FileArchive,
  LayoutGrid,
  Loader2,
  Search,
  Star,
  Table,
  X,
} from 'lucide-react'

/** Fetch every candidate in one page (the grid scrolls, no paging). */
/** Sticky-left column widths (px). Cumulative offsets are derived below. */
const W = { rank: 48, like: 42, file: 242 }
const OFF = {
  rank: 0,
  // The pinned like/file columns start 2px BEFORE the boundary and are 2px
  // wider: they overlap the previous pinned column so WebKitGTK's imprecise
  // sticky positioning can never open a background seam between them. The
  // visible separator is each overlapping cell's border-l (later siblings
  // paint on top).
  like: W.rank - 2,
  file: W.rank + W.like - 4,
}

/**
 * Every rubric score is shown on one common scale: **out of 100** (see
 * `@/lib/scores` for the arithmetic). A question's weight is not its ceiling —
 * it is only its share of the Average — so a column reads as a grade
 * (`FB2  82.8`) rather than as points-in-a-bucket (`FB2  4.1 /5`, which used
 * to make every column a different scale). The Average is the weighted average of
 * those grades, so it too reads out of 100 whatever the weights add up to.
 */

/** Column headers are just the question's name — the weight is a multiplier
 * into the Average, not a ceiling, and it is explained once in the Scoring &
 * rubric dialog rather than repeated on every column. It stays available on
 * hover via `scoreTip`. */

/** Cell tooltip: the grade's scale, plus the weight behind the Average. */
function scoreTip(label: string, question: string, weight: number): string {
  return `${label} — ${question} · out of 100${
    Math.abs(weight) > 1e-9 ? ` · weight ×${fmtWeight(weight)}` : ''
  }`
}

/** Sort-direction glyph for the active column (neutral ↕ when inactive). */
function SortIcon({ active, dir }: { active: boolean; dir: boolean }) {
  if (!active) return <span className="size-3 opacity-30">↕</span>
  return dir ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />
}

/** Sortable column header (click to toggle asc/desc). */ function SortHeader({
  label,
  sortKey,
  align = 'left',
  active,
  dir,
  onSort,
  className,
  style,
  title,
}: {
  label: string
  sortKey: string | null
  align?: 'left' | 'right'
  active: boolean
  dir: boolean
  onSort: (key: string | null) => void
  className?: string
  style?: React.CSSProperties
  title?: string
}) {
  return (
    <th
      style={style}
      className={cn(
        'sticky top-0 z-20 border-b border-l bg-muted px-2 py-1.5 font-semibold text-muted-foreground',
        align === 'right' ? 'text-right' : 'text-left',
        'min-w-[4.5rem] break-words',
        className
      )}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={cn(
          'flex w-full items-center gap-1 hover:text-foreground',
          align === 'right' && 'justify-end'
        )}
        title={title ?? `Sort by ${label}`}
      >
        <span className="leading-tight">{label}</span>
        <SortIcon active={active} dir={dir} />
      </button>
    </th>
  )
}

/** Score chip inside a card row ("FB1: 62"); click sorts by that column. */
function CardChip({
  label,
  value,
  neg,
  title,
  onClick,
}: {
  label: string
  value: string
  neg?: boolean
  title: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={cn(
        'flex shrink-0 items-center gap-1 rounded bg-muted px-1.5 py-0.5 tabular-nums hover:bg-accent',
        neg ? 'text-rose-600' : 'text-muted-foreground'
      )}
    >
      <span className="font-medium">{label}:</span>
      <span>{value}</span>
    </button>
  )
}

/**
 * Right pane: a single spreadsheet-style grid of every CV. Columns are rank, the
 * feedback actions (like / move up / move down), file name, total points, and one
 * column per rubric question. Click a header to sort ascending/descending; use the
 * "Columns" menu to show/hide columns. Horizontal + vertical scroll as needed.
 */
export function CandidatesPane({
  jobId,
  detail,
  selectedId,
  onSelect,
  onExport,
  query,
  onQueryChange,
  sortBy,
  onSortByChange,
  sortDir,
  onSortDirChange,
}: {
  jobId: string
  detail: JobDetail
  selectedId: string | null
  onSelect: (candidateId: string) => void
  /** Replace this pane with the export view (setup pane stays visible). */
  onExport: () => void
  query: string
  onQueryChange: (q: string) => void
  sortBy: string | null
  onSortByChange: (s: string | null) => void
  sortDir: boolean
  onSortDirChange: (asc: boolean) => void
}) {
  const [input, setInput] = useState(query)
  const debouncedQuery = useDebouncedValue(input, 300)
  const { t } = useTranslation()
  const rankJob = useRankJob(jobId)
  // `ranking`, not `running`: the import auto-start embeds ad-less jobs in the
  // background (no ad = nothing to rank), which must not spin this button.
  const rankRunning = rankJob.state?.ranking === true
  const updateJob = useUpdateJob(jobId)
  const emptyCandidates = useEmptyCandidates(jobId)
  const importActive = !!useImportStore(s => s.progress[jobId])?.active
  const { data: likedIds } = useLikedCandidateIds(jobId)
  const like = useLikeCandidate(jobId, detail.triqui)
  const job = detail.job
  const triqui = detail.triqui

  // Criteria are numbered (C1, C2, …) and the two feedback rows (FB1, FB2) so
  // long criterion/feedback texts stay compact in the results sheet. The
  // cascade's stage columns get the same treatment: their long labels
  // ("Job fit (judge)") would otherwise wrap the header over three lines.
  const criterionNumbers = useMemo(() => {
    const m = new Map<string, number>()
    detail.criteria.forEach((c, i) => m.set(c.id, i + 1))
    return m
  }, [detail.criteria])
  const critLabel = (q: RubricQuestion) =>
    q.kind === 'Criterion'
      ? `C${criterionNumbers.get(q.question_id) ?? '?'}`
      : q.kind === 'FeedbackJob'
        ? 'FB1'
        : q.kind === 'FeedbackProfile'
          ? 'FB2'
          : q.kind === 'Scout'
            ? 'SCO'
            : q.kind === 'Ad'
              ? 'EMB'
              : q.kind === 'AdJudge'
                ? 'JUD'
                : q.label

  // Two independent quality axes (scoring v3, gui/design/app1/cascade-simple.md):
  // which embedding space this row's criteria and feedback scores come from, and
  // what backs the job-ad slot of its total. They were one conflated 'Basis' chip
  // once, which made the sort key hard to explain — a row can be scout-space on its
  // criteria and judge-backed on its ad at the same time.
  const spaceChip = (
    space: GradeSpace | null | undefined
  ): { label: string; title: string } =>
    space === 'heavy'
      ? {
          label: 'FULL',
          title: 'Criteria and feedback scored with the selected embedder',
        }
      : space === 'scout'
        ? {
            label: 'SCO',
            title:
              'Criteria and feedback scored with the fast scout — the embedder has not reached this CV yet',
          }
        : { label: '–', title: 'Nothing has graded this candidate yet' }

  const sourceChip = (
    src: AdSource | null | undefined
  ): { label: string; title: string } =>
    src === 'judge'
      ? { label: 'JUD', title: 'Job-fit slot backed by the judge' }
      : src === 'heavy'
        ? { label: 'EMB', title: 'Job-fit slot backed by the embedder' }
        : src === 'scout'
          ? { label: 'SCO', title: 'Job-fit slot backed by the scout' }
          : { label: '–', title: 'No stage has graded this candidate yet' }

  // Feedback staging store (liked set for instant re-rank).
  const likes = useFeedbackStore(selectLikes(jobId))
  const init = useFeedbackStore(s => s.init)

  const total = detail.cv_imported + detail.cv_failed

  // Backend-derived enablement (same source as the title bar's Sort): the
  // serve owns sort_allowed + the reason code; the gui i18n-maps it. The
  // backend rank starts the engine on demand, so a cold engine never
  // disables Re-rank — `engine-not-ready` still allows the click.
  const { data: uiState } = useJobUiState(jobId)
  const sortAllowed = uiState ? uiState.sort_allowed : detail.cv_imported >= 1
  const rerankReason = !sortAllowed
    ? t(`rank.reason.${uiState?.reason ?? 'no-cvs'}`)
    : t('rank.rerankTitle')
  const rerankDirty = uiState?.rerank_dirty ?? false

  // Backend-decided guidance (thin-gui: the serve owns the codes, the gui
  // maps them). The convergence nudge — embed-prefill done, ad is the only
  // missing step — wins over the always-on "paste the ad" call-to-action
  // until dismissed (dismissal is session-only view state). With no CVs at
  // all the pane shows just the bare status lines instead of an empty grid.
  const nudgeDismissed = useUIStore(s => s.dismissedNudges.includes(jobId))
  const dismissNudge = useUIStore(s => s.dismissNudge)
  const nudgeActive = uiState?.nudge === 'add-job-ad' && !nudgeDismissed
  const showAdCta = !nudgeActive && uiState?.reason === 'no-jobad'
  const noCvs = detail.cv_imported === 0
  // Same predicate the backend uses for `ad_set`.
  const adSet = Boolean(job.job_ad_text?.trim())

  // The ad is a file now, so "add the ad" means opening that file in the user's
  // editor. `useJobAdFile` also resolves/creates it, so this works on a job that
  // has never had an ad.
  const { data: adFile } = useJobAdFile(jobId)
  const handleAddJobAd = async () => {
    const p = adFile?.path
    if (!p) return
    try {
      await openPath(p)
    } catch (e) {
      toast.error(`Could not open the job ad: ${errorMessage(e)}`)
    }
  }

  // Mirror the debounced query up to the parent (source of truth for the dialog walk).
  useEffect(() => {
    if (debouncedQuery !== query) {
      onQueryChange(debouncedQuery)
    }
  }, [debouncedQuery, query, onQueryChange])

  // Seed the feedback stage from the current triqui + liked set (keeps in-flight edits).
  useEffect(() => {
    init(
      jobId,
      triqui.map(s => s.id),
      likedIds ?? []
    )
  }, [jobId, triqui, likedIds, init])

  // --- FLIP animation refs (the layout effect is attached after `items` below) ---
  const rowEls = useRef(new Map<string, HTMLTableRowElement>())
  const prevRects = useRef(new Map<string, DOMRect>())

  // Column visibility (file name + total + each rubric question). Action + rank stay pinned.
  const [hidden, setHidden] = useState<Record<string, boolean>>({})

  // Sheet layout: classic spreadsheet columns, or one card per candidate for
  // narrow windows (the card carries every visible score).
  const [view, setView] = useState<'table' | 'cards'>('table')

  // Column show/hide menu: opens on right-click of the table header, anchored
  // at the cursor (via a fixed-position invisible dropdown trigger).
  const [colsMenuPos, setColsMenuPos] = useState<{
    x: number
    y: number
  } | null>(null)

  const handleSearchChange = (value: string) => {
    setInput(value)
  }

  const pageSize = job.triqui_size
  const [page, setPage] = useState(0)

  // Reset to first page when page size changes
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPage(0)
  }, [pageSize])

  const rubric: RubricQuestion[] = detail.rubric

  // All rows, sorted server-side by the chosen column + direction.
  const { data: pageData, isFetching } = useCandidatesPage(
    jobId,
    page * pageSize,
    pageSize,
    debouncedQuery,
    sortBy,
    sortDir,
    rerankDirty
  )
  const items = useMemo(() => pageData?.items ?? [], [pageData])
  const browserTotal = pageData?.total ?? 0

  // Per-row question_id -> {points, max} map. The weight is kept next to the
  // points so each cell can be rescaled to the shared 0..100 grade (the rubric
  // column list carries weights too, but the stored row is what was actually
  // scored — a stage that has not run yet has no entry here at all).
  const scoreByRow = useMemo(() => {
    const m = new Map<string, Map<string, { pts: number; max: number }>>()
    for (const it of items) {
      const sm = new Map<string, { pts: number; max: number }>()
      for (const s of it.scores ?? [])
        sm.set(s.question_id, { pts: s.points, max: s.max_points })
      m.set(it.id, sm)
    }
    return m
  }, [items])

  // --- FLIP animation: glide rows from their previous position to the new one whenever
  // the (server-sorted) order changes, so up/down / like / sort moves are visible. ---
  useLayoutEffect(() => {
    const newRects = new Map<string, DOMRect>()
    for (const row of items) {
      const el = rowEls.current.get(row.id)
      if (!el) continue
      const rect = el.getBoundingClientRect()
      newRects.set(row.id, rect)
      const prev = prevRects.current.get(row.id)
      if (prev) {
        const dy = prev.top - rect.top
        const dx = prev.left - rect.left
        if (dy || dx) {
          el.style.transition = 'none'
          el.style.transform = `translate(${dx}px, ${dy}px)`
          // Force reflow so the start transform is committed before we animate.
          void el.offsetWidth
          requestAnimationFrame(() => {
            el.style.transition =
              'transform 450ms cubic-bezier(0.22, 1, 0.36, 1)'
            el.style.transform = ''
          })
        }
      }
    }
    prevRects.current = newRects
  }, [items])

  const handleSort = (key: string | null) => {
    if (sortBy === key) {
      onSortDirChange(!sortDir)
    } else {
      onSortByChange(key)
      onSortDirChange(key === '__name')
    }
  }

  // Re-rank with the current vectors (manual re-rank after e.g. editing weights).
  const handleRerank = () => rankJob.mutate()

  // Results CSV: pure data export (no CV files) via the backend command.
  const exportCsv = useExportResultsCsv()
  const handleExportCsv = async () => {
    const safe = (job.title || 'job')
      .replace(/[^a-z0-9-_]+/gi, '_')
      .slice(0, 60)
    const path = await saveFileDialog({
      defaultPath: `${safe}.csv`,
      filters: [{ name: 'CSV', extensions: ['csv'] }],
    })
    if (!path) return
    exportCsv.mutate(
      { jobId, outPath: path },
      {
        onSuccess: s => toast.success(`Exported ${s.rows} results`),
        onError: e => toast.error(`Export failed: ${errorMessage(e)}`),
      }
    )
  }

  // Fixed table layout honors the pinned columns' style widths exactly (auto
  // layout ignores them, desyncing the sticky `left` offsets and reopening the
  // background seams between pinned cells). Fixed layout ignores per-cell
  // min-width, so the 72px floor for Average + rubric columns lives on the
  // table itself — narrow panes scroll horizontally instead of crushing them.
  const scoreColCount =
    (hidden.average ? 0 : 1) +
    (hidden['__space'] ? 0 : 1) +
    rubric.filter(q => !hidden[q.question_id]).length
  // The Average column needs no ceiling any more: every score — the Average
  // included — is read on the same 0..100 scale (`per100`). The rubric's
  // capacity (Σ |weights|) only ever mattered as that per-column denominator,
  // and each cell now divides by its own weight. Stage columns (scout, judge)
  // are graded on the ad's scale and stay display-only, as the backend decides.
  const tableStyle: React.CSSProperties | undefined =
    view === 'cards'
      ? undefined
      : scoreColCount > 0
        ? { minWidth: W.rank + W.like + W.file + 72 * scoreColCount }
        : // Every score column hidden: w-full would stretch the pinned
          // columns proportionally — pin the table at its natural width.
          { width: W.rank + W.like + W.file }

  return (
    <section className="flex min-h-[20rem] flex-col gap-3 rounded-lg border bg-card p-4 md:h-full md:flex-1 md:min-w-0 md:overflow-hidden">
      <div className="flex items-center gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Candidates
        </h2>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex items-center gap-0.5 rounded-md border p-0.5">
            <Button
              variant="ghost"
              size="icon-sm"
              className={cn('size-7', view === 'table' && 'bg-accent')}
              onClick={() => setView('table')}
              title="Table view"
            >
              <Table className="size-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              className={cn('size-7', view === 'cards' && 'bg-accent')}
              onClick={() => setView('cards')}
              title="Card view"
            >
              <LayoutGrid className="size-4" />
            </Button>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" title="Export">
                <Download className="size-4" />
                Export
                <ChevronDown className="size-3.5 text-muted-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onExport}>
                <FileArchive className="size-4" />
                <span className="min-w-0">
                  <span className="block">Job bundle…</span>
                  <span className="block text-xs text-muted-foreground">
                    Portable .triquijob with CV files
                  </span>
                </span>
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={handleExportCsv}
                disabled={!job.last_ranked_at || exportCsv.isPending}
                title={
                  job.last_ranked_at
                    ? 'Download the ranked results as a CSV file'
                    : 'Rank the candidates first'
                }
              >
                <Table className="size-4" />
                <span className="min-w-0">
                  <span className="block">Results as CSV…</span>
                  <span className="block text-xs text-muted-foreground">
                    All ranked candidates, one column per question
                  </span>
                </span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            size="sm"
            onClick={handleRerank}
            disabled={!sortAllowed || rankJob.isPending || rankRunning}
            title={rerankReason}
          >
            {rankJob.isPending || rankRunning ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ArrowUp className="size-4" />
            )}{' '}
            {rankRunning ? 'Ranking…' : 'Re-rank'}
          </Button>
        </div>
      </div>

      {/* Backend-derived hold state (libsyscap): the ranking is alive but the
          waves are paused because the machine is nearly out of memory. The
          backend owns the decision — the gui only maps the `holding` reason
          code to a visible notice; it clears itself when the verdict lifts. */}
      {rankJob.state?.reason === 'holding' && (
        <div
          role="status"
          className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-400"
        >
          <Loader2 className="size-3.5 shrink-0 animate-spin" />
          {t('rank.holding')}
        </div>
      )}

      {/* Backend-decided nudge (`job_ui_state.nudge = add-job-ad`): the ad-less
          embed-prefill run converged — all the background work is done, so the
          job ad is the only missing step. Non-blocking; dismiss lasts the
          session (view state stays client-side). */}
      {nudgeActive && (
        <div
          role="status"
          className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-700 dark:text-emerald-400"
        >
          <CheckCircle2 className="size-3.5 shrink-0" />
          <span className="min-w-0 flex-1">
            {t('rank.nudge.embedded', { n: detail.cv_imported })}
          </span>
          <Button size="sm" className="h-6 text-xs" onClick={handleAddJobAd}>
            {t('rank.nudge.action')}
          </Button>
          <button
            type="button"
            onClick={() => dismissNudge(jobId)}
            className="rounded p-0.5 hover:bg-foreground/10"
            aria-label="Dismiss"
            title="Dismiss"
          >
            <X className="size-3.5" />
          </button>
        </div>
      )}
      {/* Persistent, quieter sibling: CVs are in but there is no ad to rank
          against (`no-jobad`). Hidden while the convergence banner shows. */}
      {showAdCta && (
        <div
          role="status"
          className="flex items-center gap-2 rounded-md border bg-muted/40 px-3 py-1.5 text-xs text-muted-foreground"
        >
          <span className="min-w-0 flex-1">
            {t('rank.cta.pasteAd', { n: detail.cv_imported })}
          </span>
          <Button
            variant="outline"
            size="sm"
            className="h-6 text-xs"
            onClick={handleAddJobAd}
          >
            {t('rank.nudge.action')}
          </Button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <label className="flex items-center gap-1">
          Top
          <Input
            type="number"
            min={1}
            step={1}
            defaultValue={job.triqui_size}
            onBlur={e => {
              const n = Math.max(1, Math.floor(Number(e.target.value) || 1))
              e.target.value = String(n)
              if (n !== job.triqui_size) {
                updateJob.mutate({
                  title: job.title,
                  jobAdText: job.job_ad_text ?? null,
                  triquiSize: n,
                })
              }
            }}
            className="h-7 w-20 text-xs"
          />
        </label>
        {!importActive && total > 0 && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto h-7 text-xs text-muted-foreground hover:text-destructive"
              >
                <Eraser className="size-3.5" />
                Empty candidates
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Empty all candidates?</AlertDialogTitle>
                <AlertDialogDescription>
                  This removes every imported CV for this job so you can
                  re-import a fresh set. Your job ad and criteria are kept.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-destructive text-white hover:bg-destructive/90"
                  onClick={() =>
                    emptyCandidates.mutate(undefined, {
                      onSuccess: () =>
                        toast.success(
                          'Cleared all candidates — ready to re-import'
                        ),
                      onError: e =>
                        toast.error(
                          `Could not empty candidates: ${errorMessage(e)}`
                        ),
                    })
                  }
                >
                  Empty candidates
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>

      {/* border-separate (not collapse): collapsed borders break sticky-left
          columns — scrolled-through cells peek through the seams. table-fixed
          keeps every column at its styled width so the pinned cells always
          tile without gaps under the sticky `left` offsets. */}
      <div className="relative min-w-0 flex-1 overflow-auto rounded-md border">
        {noCvs ? (
          <div className="flex h-full min-h-56 flex-col items-center justify-center gap-2 p-6 text-center">
            {noCvs && (
              <p className="text-sm font-medium text-muted-foreground">
                {t('candidates.empty.title')}
              </p>
            )}
            {!adSet && (
              <p className="text-sm font-medium text-muted-foreground">
                {t('candidates.empty.noJobAd')}
              </p>
            )}
          </div>
        ) : (
          <table
            style={tableStyle}
            className="w-full table-fixed border-separate [border-spacing:0] text-xs"
          >
            <thead
              onContextMenu={e => {
                // Right-click a column header to show/hide columns.
                e.preventDefault()
                setColsMenuPos({ x: e.clientX, y: e.clientY })
              }}
            >
              <tr>
                {/* Pinned-left identity + action columns */}
                <th
                  style={{ left: OFF.rank, width: W.rank }}
                  className="sticky top-0 z-30 border-b bg-muted px-1 py-1.5 text-right font-semibold text-muted-foreground"
                >
                  <button
                    type="button"
                    onClick={() => handleSort('__rank')}
                    className="flex w-full items-center justify-end gap-1 hover:text-foreground"
                    title="Sort by rank"
                  >
                    #
                    <SortIcon active={sortBy === '__rank'} dir={sortDir} />
                  </button>
                </th>
                <th
                  style={{ left: OFF.like, width: W.like }}
                  className="sticky top-0 z-30 border-b border-l bg-muted px-1 py-1.5 text-center font-semibold text-muted-foreground"
                  title="Like"
                >
                  ★
                </th>
                {view === 'table' ? (
                  <>
                    {!hidden.filename && (
                      <SortHeader
                        label="File name"
                        sortKey="__name"
                        active={sortBy === '__name'}
                        dir={sortDir}
                        onSort={handleSort}
                        style={{ left: OFF.file, width: W.file }}
                        className="z-30"
                      />
                    )}

                    {/* Data columns */}
                    {!hidden.average && (
                      <SortHeader
                        label="Average"
                        title="Weighted average of every score in this row (out of 100)"
                        sortKey={null}
                        align="right"
                        active={sortBy === null}
                        dir={sortDir}
                        onSort={handleSort}
                      />
                    )}
                    {!hidden['__space'] && (
                      <SortHeader
                        label="Quality"
                        sortKey="__space"
                        active={sortBy === '__space'}
                        dir={sortDir}
                        onSort={handleSort}
                        title="Sort by how the row was scored: the full embedder's space first, then the scout's, and by the total inside each"
                      />
                    )}
                    {!hidden['__source'] && (
                      <SortHeader
                        label="Fit"
                        sortKey="__source"
                        active={sortBy === '__source'}
                        dir={sortDir}
                        onSort={handleSort}
                        title="Sort by what backs the job-fit slot — judge, then embedder, then scout"
                      />
                    )}
                    {rubric
                      .filter(q => !hidden[q.question_id])
                      .map(q => (
                        <SortHeader
                          key={q.question_id}
                          label={critLabel(q)}
                          sortKey={q.question_id}
                          align="right"
                          active={sortBy === q.question_id}
                          dir={sortDir}
                          onSort={handleSort}
                          title={`Sort by ${critLabel(q)} — ${q.label}`}
                        />
                      ))}
                  </>
                ) : (
                  <SortHeader
                    label="File name"
                    sortKey="__name"
                    active={sortBy === '__name'}
                    dir={sortDir}
                    onSort={handleSort}
                  />
                )}
              </tr>
            </thead>
            <tbody>
              {items.map(row => {
                const active = row.id === selectedId
                const liked = !!likes[row.id]
                const rowScores = scoreByRow.get(row.id)
                return (
                  <tr
                    key={row.id}
                    ref={el => {
                      if (el) rowEls.current.set(row.id, el)
                      else rowEls.current.delete(row.id)
                    }}
                    className={cn('hover:bg-accent/40', active && 'bg-accent')}
                  >
                    <td
                      style={{ left: OFF.rank, width: W.rank }}
                      className={cn(
                        'sticky z-10 border-b px-1 py-1 text-right tabular-nums text-muted-foreground',
                        active ? 'bg-accent' : 'bg-card'
                      )}
                    >
                      {row.rank != null ? row.rank : '–'}
                    </td>
                    <td
                      style={{ left: OFF.like, width: W.like }}
                      className={cn(
                        'sticky z-10 border-b border-l px-1 py-1 text-center',
                        active ? 'bg-accent' : 'bg-card'
                      )}
                    >
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="h-6 w-6"
                        title={liked ? 'Unlike' : 'Like'}
                        onClick={() => like(row.id)}
                      >
                        <Star
                          className={cn(
                            'size-3.5',
                            liked
                              ? 'fill-amber-400 text-amber-500'
                              : 'text-muted-foreground'
                          )}
                        />
                      </Button>
                    </td>
                    {!hidden.filename && view === 'table' && (
                      <td
                        style={{ left: OFF.file, width: W.file }}
                        className={cn(
                          'sticky z-10 border-b border-l px-2 py-1',
                          active ? 'bg-accent' : 'bg-card'
                        )}
                      >
                        <button
                          onClick={() => onSelect(row.id)}
                          className="flex w-full min-w-0 items-center gap-1 text-left"
                          title={row.file_name}
                        >
                          <span className="truncate font-medium">
                            {row.file_name}
                          </span>
                          {row.extraction_status === 'Failed' && (
                            <Badge
                              variant="outline"
                              className="shrink-0 text-rose-600"
                            >
                              failed
                            </Badge>
                          )}
                        </button>
                      </td>
                    )}

                    {view === 'cards' ? (
                      <td
                        className={cn(
                          'border-b px-2 py-1',
                          active && 'bg-accent'
                        )}
                      >
                        <div className="flex min-w-0 flex-col items-start gap-1">
                          <div className="flex min-w-0 max-w-full items-center gap-1">
                            <button
                              onClick={() => onSelect(row.id)}
                              className="min-w-0 truncate text-left font-medium hover:underline"
                              title={row.file_name}
                            >
                              {row.file_name}
                            </button>
                            {row.extraction_status === 'Failed' && (
                              <Badge
                                variant="outline"
                                className="shrink-0 text-rose-600"
                              >
                                failed
                              </Badge>
                            )}
                          </div>
                          <div className="flex flex-wrap items-center gap-1">
                            {!hidden.average && (
                              <CardChip
                                label="Average"
                                value={fmtScore(
                                  per100(row.points, row.points_max)
                                )}
                                title="Weighted average of every score in this row (out of 100)"
                                onClick={() => handleSort(null)}
                              />
                            )}
                            {!hidden['__space'] && (
                              <CardChip
                                label="Quality"
                                value={spaceChip(row.grade_space).label}
                                title={spaceChip(row.grade_space).title}
                                onClick={() => handleSort('__space')}
                              />
                            )}
                            {!hidden['__source'] && (
                              <CardChip
                                label="Fit"
                                value={sourceChip(row.ad_source).label}
                                title={sourceChip(row.ad_source).title}
                                onClick={() => handleSort('__source')}
                              />
                            )}
                            {rubric
                              .filter(q => !hidden[q.question_id])
                              .map(q => {
                                const s = rowScores?.get(q.question_id)
                                const g = per100(s?.pts, s?.max)
                                return (
                                  <CardChip
                                    key={q.question_id}
                                    label={critLabel(q)}
                                    value={fmtScore(g)}
                                    neg={(g ?? 0) < 0}
                                    title={scoreTip(
                                      critLabel(q),
                                      q.label,
                                      q.max_points
                                    )}
                                    onClick={() => handleSort(q.question_id)}
                                  />
                                )
                              })}
                          </div>
                        </div>
                      </td>
                    ) : (
                      <>
                        {!hidden.average && (
                          <td className="border-b border-l px-2 py-1 text-right tabular-nums">
                            {fmtScore(per100(row.points, row.points_max))}
                          </td>
                        )}
                        {!hidden['__space'] && (
                          <td
                            title={spaceChip(row.grade_space).title}
                            className="border-b border-l px-2 py-1 text-center text-xs font-semibold text-muted-foreground"
                          >
                            {spaceChip(row.grade_space).label}
                          </td>
                        )}
                        {!hidden['__source'] && (
                          <td
                            title={sourceChip(row.ad_source).title}
                            className="border-b border-l px-2 py-1 text-center text-xs font-semibold text-muted-foreground"
                          >
                            {sourceChip(row.ad_source).label}
                          </td>
                        )}
                        {rubric
                          .filter(q => !hidden[q.question_id])
                          .map(q => {
                            const s = rowScores?.get(q.question_id)
                            const g = per100(s?.pts, s?.max)
                            const neg = (g ?? 0) < 0
                            return (
                              <td
                                key={q.question_id}
                                title={scoreTip(
                                  critLabel(q),
                                  q.label,
                                  q.max_points
                                )}
                                className={cn(
                                  'border-b border-l px-2 py-1 text-right tabular-nums',
                                  neg ? 'text-rose-600' : 'text-foreground'
                                )}
                              >
                                {fmtScore(g)}
                              </td>
                            )
                          })}
                      </>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}

        {!noCvs && items.length === 0 && (
          <p className="py-10 text-center text-sm text-muted-foreground">
            {isFetching
              ? 'Loading…'
              : detail.cv_imported > 0
                ? 'No CVs match your search.'
                : 'Import CVs, then Re-rank to see candidates.'}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={input}
            onChange={e => handleSearchChange(e.target.value)}
            placeholder="Search CV content…"
            className="h-8 pl-7 text-xs"
          />
        </div>
      </div>

      <Paginator
        page={page}
        pageSize={pageSize}
        total={browserTotal}
        onPageChange={setPage}
      />

      {/* Column show/hide — opens on right-click of the table header. */}
      <DropdownMenu
        open={colsMenuPos !== null}
        onOpenChange={open => {
          if (!open) setColsMenuPos(null)
        }}
      >
        <DropdownMenuTrigger asChild>
          <span
            aria-hidden
            className="pointer-events-none fixed"
            style={{ left: colsMenuPos?.x ?? 0, top: colsMenuPos?.y ?? 0 }}
          />
        </DropdownMenuTrigger>
        <DropdownMenuContent className="max-h-80 overflow-auto">
          <DropdownMenuLabel>Show columns</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            checked={!hidden.filename}
            onCheckedChange={v => setHidden(h => ({ ...h, filename: !v }))}
          >
            File name
          </DropdownMenuCheckboxItem>
          <DropdownMenuCheckboxItem
            checked={!hidden.average}
            onCheckedChange={v => setHidden(h => ({ ...h, mean: !v }))}
          >
            Average
          </DropdownMenuCheckboxItem>
          <DropdownMenuCheckboxItem
            checked={!hidden['__space']}
            onCheckedChange={v => setHidden(h => ({ ...h, __space: !v }))}
          >
            Quality
          </DropdownMenuCheckboxItem>
          <DropdownMenuCheckboxItem
            checked={!hidden['__source']}
            onCheckedChange={v => setHidden(h => ({ ...h, __source: !v }))}
          >
            Fit
          </DropdownMenuCheckboxItem>
          {rubric.map(q => {
            const short = critLabel(q)
            return (
              <DropdownMenuCheckboxItem
                key={q.question_id}
                checked={!hidden[q.question_id]}
                onCheckedChange={v =>
                  setHidden(h => ({ ...h, [q.question_id]: !v }))
                }
              >
                {short === q.label ? q.label : `${short} — ${q.label}`}
              </DropdownMenuCheckboxItem>
            )
          })}
        </DropdownMenuContent>
      </DropdownMenu>
    </section>
  )
}
