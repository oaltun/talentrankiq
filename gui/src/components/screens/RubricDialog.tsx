import { useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useDeleteCriterion, useUpdateCriterion } from '@/services/criteria'
import { useUpdateJobWeights } from '@/services/jobs'
import { scoringFormula } from '@/lib/scores'
import type { Criterion, JobDetail } from '@/lib/tauri-bindings'
import { X } from 'lucide-react'

function clampPoints(v: string): number {
  return Math.max(0, Number(v) || 0)
}

function clampInt(v: string): number {
  return Math.max(0, Math.floor(Number(v) || 0))
}

/**
 * One badge-labelled weight row of the rubric (AD / FB1 / FB2). Blur commits
 * all three weights together — `update_job_weights` always writes the full
 * triple, so a single shared commit avoids losing a sibling draft.
 */
function WeightRow({
  badge,
  label,
  value,
  onChange,
  onCommit,
}: {
  badge: string
  label: string
  value: string
  onChange: (v: string) => void
  onCommit: () => void
}) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="flex size-6 shrink-0 items-center justify-center rounded-md bg-muted text-xs text-muted-foreground"
        title={badge}
      >
        {badge}
      </span>
      <span className="flex-1 text-sm text-muted-foreground">{label}</span>
      <Input
        type="number"
        min={0}
        step={1}
        value={value}
        onChange={e => onChange(e.target.value)}
        onBlur={onCommit}
        className="w-20"
        title="weight in the Average"
      />
    </div>
  )
}

/** Criterion row inside the rubric dialog: text + weight + delete, blur-commit. */
function RubricCriterionRow({
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
  const [weight, setWeight] = useState(String(criterion.weight))

  const commit = () => {
    const t = text.trim()
    if (!t) {
      // A criterion without text is meaningless — remove it, matching the
      // Setup pane composer's empty-means-discard behavior.
      deleteCriterion.mutate(criterion.id)
      toast('Criterion deleted')
      return
    }
    if (t !== criterion.text || clampPoints(weight) !== criterion.weight) {
      updateCriterion.mutate({
        id: criterion.id,
        text: t,
        weight: clampPoints(weight),
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
        onBlur={commit}
        placeholder="criterion"
        className="h-[76px] min-h-0 flex-1 resize-none field-sizing-fixed"
      />
      <Input
        type="number"
        min={0}
        step={0.1}
        value={weight}
        onChange={e => setWeight(e.target.value)}
        onBlur={commit}
        className="w-20"
        title="weight in the Average"
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
 * Job-scoped rubric editor: every "max points" weight in one place (job ad,
 * per-criterion + shared criterion default, the two feedback rows) instead of
 * inline inputs across the panes. Opened from the Workspace header icon and
 * the Setup pane's scoring summary. All fields auto-save on blur; weight
 * edits are ranking inputs, so the triqui goes stale until re-rank.
 */
export function RubricDialog({
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
  const updateJobWeights = useUpdateJobWeights(jobId)
  const updateCriterion = useUpdateCriterion(jobId)
  const job = detail.job

  const [ad, setAd] = useState(String(job.ad_weight ?? 50))
  const [fbJob, setFbJob] = useState(String(job.fb_job_weight ?? 10))
  const [fbProf, setFbProf] = useState(String(job.fb_profile_weight ?? 10))
  const [bulk, setBulk] = useState('10')
  // Re-sync the drafts when the server values change (our own commits come
  // back clamped).
  const [stored, setStored] = useState({
    ad: job.ad_weight,
    fbJob: job.fb_job_weight,
    fbProf: job.fb_profile_weight,
  })
  if (
    stored.ad !== job.ad_weight ||
    stored.fbJob !== job.fb_job_weight ||
    stored.fbProf !== job.fb_profile_weight
  ) {
    setStored({
      ad: job.ad_weight,
      fbJob: job.fb_job_weight,
      fbProf: job.fb_profile_weight,
    })
    setAd(String(job.ad_weight ?? 50))
    setFbJob(String(job.fb_job_weight ?? 10))
    setFbProf(String(job.fb_profile_weight ?? 10))
  }

  // `update_job_weights` writes the full triple — send the current drafts of
  // all three so an uncommitted sibling isn't lost.
  const commitWeights = () =>
    updateJobWeights.mutate({
      adWeight: clampInt(ad),
      fbJobWeight: clampInt(fbJob),
      fbProfileWeight: clampInt(fbProf),
    })

  const applyBulk = () => {
    const n = clampPoints(bulk)
    for (const c of detail.criteria) {
      updateCriterion.mutate({ id: c.id, text: c.text, weight: n })
    }
    toast(`Applied ${n} points to ${detail.criteria.length} criteria`)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] flex-col overflow-hidden sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Scoring &amp; rubric</DialogTitle>
          <DialogDescription>
            Each weight is how much that question contributes to the Average.
            Saved as you edit; re-rank to apply.
          </DialogDescription>
        </DialogHeader>

        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
          <section
            className="flex flex-col gap-1.5 rounded-md bg-muted/40 p-2.5"
            aria-label="How scoring works"
          >
            <span className="text-xs font-medium">
              How a score becomes the Average
            </span>
            <p className="text-xs text-muted-foreground">
              Every score in the results sheet is graded <b>out of 100</b> — how
              well that CV answers that question. A weight is not a ceiling; it
              is the multiplier that carries that grade into the Average. The
              Average is their weighted mean, so it reads out of 100 too
              whatever the weights add up to:
            </p>
            <p className="font-mono text-xs text-foreground">
              {scoringFormula(job, detail.criteria)}
              <span className="text-muted-foreground"> (each ÷ 100)</span>
            </p>
            <p className="text-xs text-muted-foreground">
              The judge row&rsquo;s grade is the tournament win-rate, so its 70
              means &ldquo;wins 70% of its comparisons&rdquo;.
            </p>
          </section>

          <section className="flex flex-col gap-1.5" aria-label="Weights">
            <WeightRow
              badge="AD"
              label="Job ad similarity"
              value={ad}
              onChange={setAd}
              onCommit={commitWeights}
            />
            <WeightRow
              badge="FB1"
              label="Feedback from this job"
              value={fbJob}
              onChange={setFbJob}
              onCommit={commitWeights}
            />
            <WeightRow
              badge="FB2"
              label="Feedback from other jobs"
              value={fbProf}
              onChange={setFbProf}
              onCommit={commitWeights}
            />
          </section>

          <section className="flex flex-col gap-1.5" aria-label="Criteria">
            <div className="flex items-center gap-2">
              <span className="flex-1 text-sm text-muted-foreground">
                Criterion weights
              </span>
              <Input
                type="number"
                min={0}
                step={0.1}
                value={bulk}
                onChange={e => setBulk(e.target.value)}
                className="w-20"
                title="weight for every criterion"
              />
              <Button
                variant="outline"
                size="sm"
                disabled={detail.criteria.length === 0}
                onClick={applyBulk}
              >
                Apply to all
              </Button>
            </div>
            {detail.criteria.map((c, i) => (
              <RubricCriterionRow
                key={c.id}
                jobId={jobId}
                criterion={c}
                index={i + 1}
              />
            ))}
            {detail.criteria.length === 0 && (
              <span className="text-xs text-muted-foreground italic">
                No criteria yet — add them in the Setup pane.
              </span>
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  )
}
