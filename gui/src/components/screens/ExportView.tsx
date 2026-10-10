import { useState } from 'react'
import { toast } from 'sonner'
import { saveFileDialog } from '@/lib/file-dialogs'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { errorMessage } from '@/lib/errors'
import type { Criterion, JobDetail } from '@/lib/tauri-bindings'
import { useExportJobBundle } from '@/services/job-bundle'
import { FileArchive, Loader2 } from 'lucide-react'

const BUNDLE_EXTENSION = 'triquijob'

function OptionRow({
  checked,
  onCheckedChange,
  title,
  description,
}: {
  checked: boolean
  onCheckedChange: (v: boolean) => void
  title: string
  description?: string
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 rounded-md px-1 py-1.5 hover:bg-accent/40">
      <Checkbox
        checked={checked}
        onCheckedChange={v => onCheckedChange(v === true)}
        className="mt-0.5"
      />
      <span className="min-w-0">
        <span className="block text-sm">{title}</span>
        {description && (
          <span className="block text-xs text-muted-foreground">
            {description}
          </span>
        )}
      </span>
    </label>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
      {children}
    </span>
  )
}

function CriterionRow({
  criterion,
  index,
  checked,
  onCheckedChange,
}: {
  criterion: Criterion
  index: number
  checked: boolean
  onCheckedChange: (v: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 rounded-md px-1 py-1.5 hover:bg-accent/40">
      <Checkbox
        checked={checked}
        onCheckedChange={v => onCheckedChange(v === true)}
        className="mt-0.5"
      />
      <span
        className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md bg-muted text-xs text-muted-foreground"
        title={`Criterion ${index}`}
      >
        {index}
      </span>
      <span className="line-clamp-2 min-w-0 flex-1 text-sm">
        {criterion.text}
      </span>
      <span
        className="mt-0.5 shrink-0 text-xs tabular-nums text-muted-foreground"
        title="max points"
      >
        {criterion.weight}
      </span>
    </label>
  )
}

/**
 * Right drawer over the job workspace: export options stay next to the Setup
 * pane so criterion labels/weights can be cross-referenced while choosing.
 * CV files are always included — without them the bundle is just names and
 * numbers — capped to the top N ranked candidates chosen below. Escape/
 * overlay-click/Cancel all close it.
 */
export function ExportView({
  jobId,
  detail,
  open,
  onOpenChange,
}: {
  jobId: string
  detail: JobDetail
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const job = detail.job
  const exportBundle = useExportJobBundle()

  const [includeAd, setIncludeAd] = useState(true)
  const [includeWeights, setIncludeWeights] = useState(true)
  const [includeResults, setIncludeResults] = useState(true)
  const [includeLikes, setIncludeLikes] = useState(true)
  const [criterionIds, setCriterionIds] = useState<string[]>([])
  const [topCount, setTopCount] = useState(String(job.triqui_size))

  const toggleCriterion = (id: string, v: boolean) =>
    setCriterionIds(ids => (v ? [...ids, id] : ids.filter(x => x !== id)))

  // Empty input = export everything; otherwise clamp to 1..cv_imported.
  const topN: number | null = (() => {
    const raw = Number.parseInt(topCount, 10)
    if (Number.isNaN(raw)) return null
    return detail.cv_imported > 0
      ? Math.max(1, Math.min(raw, detail.cv_imported))
      : Math.max(1, raw)
  })()

  const handleExport = async () => {
    const safe = (job.title || 'job')
      .replace(/[^a-z0-9-_]+/gi, '_')
      .slice(0, 60)
    const path = await saveFileDialog({
      defaultPath: `${safe}.${BUNDLE_EXTENSION}`,
      filters: [{ name: 'TalentRankIQ job', extensions: [BUNDLE_EXTENSION] }],
    })
    if (!path) return
    exportBundle.mutate(
      {
        jobId,
        options: {
          include_ad: includeAd,
          include_weights: includeWeights,
          include_results: includeResults,
          include_likes: includeLikes,
          criterion_ids: criterionIds,
          top_n: topN,
        },
        outPath: path,
      },
      {
        onSuccess: s => {
          const missing =
            s.files_missing > 0
              ? ` (${s.files_missing} source files missing)`
              : ''
          toast.success(`Exported ${s.candidates_total} CVs${missing}`)
          onOpenChange(false)
        },
        onError: e => toast.error(`Export failed: ${errorMessage(e)}`),
      }
    )
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex flex-col gap-0 sm:max-w-md">
        <SheetHeader className="border-b">
          <SheetTitle>Export job</SheetTitle>
          <SheetDescription className="truncate">{job.title}</SheetDescription>
        </SheetHeader>

        <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-4">
          <p className="text-xs text-muted-foreground">
            Select what to export.
          </p>

          <div className="flex flex-col gap-1">
            <SectionLabel>Candidates</SectionLabel>
            <div className="flex items-start justify-between gap-3 rounded-md px-1 py-1.5">
              <span className="min-w-0">
                <span className="block text-sm">Top results</span>
                <span className="block text-xs text-muted-foreground">
                  Ranked CV files, best first — out of {detail.cv_imported}{' '}
                  imported. Leave empty for all.
                </span>
              </span>
              <Input
                type="number"
                min={1}
                max={detail.cv_imported || undefined}
                step={1}
                value={topCount}
                onChange={e => setTopCount(e.target.value)}
                placeholder="all"
                className="h-8 w-20 shrink-0"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <SectionLabel>Job setup</SectionLabel>
            <OptionRow
              checked={includeAd}
              onCheckedChange={setIncludeAd}
              title="Job advertisement"
            />
            <OptionRow
              checked={includeWeights}
              onCheckedChange={setIncludeWeights}
              title="Point weights"
              description="Max points for the job ad and the selected criteria. Feedback weights are personal and stay private."
            />
          </div>

          <div className="flex flex-col gap-1">
            <SectionLabel>Additional criteria</SectionLabel>
            {detail.criteria.length === 0 ? (
              <span className="px-1 py-1 text-sm text-muted-foreground italic">
                No criteria defined.
              </span>
            ) : (
              detail.criteria.map((c, i) => (
                <CriterionRow
                  key={c.id}
                  criterion={c}
                  index={i + 1}
                  checked={criterionIds.includes(c.id)}
                  onCheckedChange={v => toggleCriterion(c.id, v)}
                />
              ))
            )}
            {detail.criteria.length > 0 && criterionIds.length === 0 && (
              <span className="px-1 pt-1 text-xs text-muted-foreground">
                Criteria are private by default — select the ones to include.
              </span>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <SectionLabel>Results &amp; feedback</SectionLabel>
            <OptionRow
              checked={includeResults}
              onCheckedChange={setIncludeResults}
              title="Ranked results"
              description="Ranks, total points and per-question scores for the included criteria."
            />
            <OptionRow
              checked={includeLikes}
              onCheckedChange={setIncludeLikes}
              title="Likes"
              description="Starred candidates. Learned feedback vectors stay private."
            />
          </div>
        </div>

        <SheetFooter className="flex-row items-center gap-2 border-t">
          <div className="flex-1" />
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleExport}
            disabled={exportBundle.isPending}
          >
            {exportBundle.isPending ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <FileArchive className="size-4" />
            )}
            Export…
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
