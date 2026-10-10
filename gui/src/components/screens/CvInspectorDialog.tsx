import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { openPath } from '@tauri-apps/plugin-opener'

import { cn } from '@/lib/utils'
import { errorMessage } from '@/lib/errors'
import { fmtScore, per100 } from '@/lib/scores'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ChunkMatchTarget, JobDetail } from '@/lib/tauri-bindings'
import { useCandidate, useCandidateNeighbors } from '@/services/candidates'
import { useChunkMatch, useRetryRuntimeStatus } from '@/services/match-lens'
import { runtimeReady, useRuntimeStatus } from '@/services/runtime'
import { useLikeCandidate } from '@/services/feedback'
import { useDebouncedValue } from '@/hooks/use-debounced-value'
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Star,
  XIcon,
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { selectLikes, useFeedbackStore } from '@/store/feedback-store'
import { toast } from 'sonner'
import { clampChunkSize } from '@/lib/match-lens'
import { MatchLensControls } from './match-lens/MatchLensControls'
import type { LensTargetKind } from './match-lens/MatchLensControls'
import { HighlightedCv } from './match-lens/HighlightedCv'

interface CvInspectorDialogProps {
  jobId: string
  detail: JobDetail
  candidateId: string | null
  query: string
  sortBy: string | null
  sortDir: boolean
  onSelect: (candidateId: string) => void
  onClose: () => void
}

export function CvInspectorDialog({
  jobId,
  detail,
  candidateId,
  query,
  sortBy,
  sortDir,
  onSelect,
  onClose,
}: CvInspectorDialogProps) {
  const { t } = useTranslation()
  const { data: candidate, isLoading } = useCandidate(candidateId)
  const { data: neighbors } = useCandidateNeighbors(
    jobId,
    candidateId,
    query,
    sortBy,
    sortDir
  )

  const likes = useFeedbackStore(selectLikes(jobId))
  const like = useLikeCandidate(jobId, detail.triqui)
  const liked = candidate ? !!likes[candidate.id] : false

  const hasPrev = !!neighbors?.prev_id
  const hasNext = !!neighbors?.next_id
  const handlePrev = () => {
    if (neighbors?.prev_id) onSelect(neighbors.prev_id)
  }
  const handleNext = () => {
    if (neighbors?.next_id) onSelect(neighbors.next_id)
  }

  // Match Lens state — local to the dialog, not persisted (v1).
  const [targetKind, setTargetKind] = useState<LensTargetKind>('jobAd')
  const [criterionId, setCriterionId] = useState<string | null>(null)
  const [sizeDraft, setSizeDraft] = useState(8)
  const [topN, setTopN] = useState(5)

  const chunkSize = useDebouncedValue(clampChunkSize(sizeDraft), 300)

  const target: ChunkMatchTarget =
    targetKind === 'jobAd' ? 'JobAd' : { Criterion: { id: criterionId ?? '' } }

  const hasText =
    !!candidate?.extracted_text && candidate.extracted_text.trim().length > 0
  const runtime = useRuntimeStatus()
  const engineReady = runtimeReady(runtime.data)
  const retryRuntime = useRetryRuntimeStatus()
  const chunkQuery = useChunkMatch(
    candidateId,
    target,
    chunkSize,
    engineReady && hasText
  )

  const runtimeStatus = runtime.data?.status
  const selectTarget = (kind: LensTargetKind, id: string | null) => {
    setTargetKind(kind)
    setCriterionId(id)
  }

  return (
    <Dialog open={!!candidateId} onOpenChange={open => !open && onClose()}>
      <DialogContent
        className="flex h-[calc(100vh-3rem)] w-[calc(100vw-3rem)] max-w-[calc(100vw-3rem)] flex-col overflow-hidden sm:max-w-[calc(100vw-3rem)]"
        showCloseButton={false}
      >
        <DialogHeader className="flex shrink-0 flex-row justify-between items-start">
          <div className="flex flex-col gap-1.5">
            <DialogTitle>
              {candidate?.file_name
                ? t('cvInspector.title', { fileName: candidate.file_name })
                : isLoading
                  ? 'Loading\u2026'
                  : t('cvInspector.title', {
                      fileName: t('cvInspector.candidate'),
                    })}
            </DialogTitle>
            <div className="flex items-center gap-2">
              {candidate?.rank != null && (
                <Badge variant="secondary">#{candidate.rank}</Badge>
              )}
              {candidate?.points != null &&
                candidate.points_max != null &&
                per100(candidate.points, candidate.points_max) != null && (
                  <Badge
                    variant="outline"
                    title="Weighted average of this CV's rubric scores, out of 100"
                  >
                    {fmtScore(per100(candidate.points, candidate.points_max))} /
                    100
                  </Badge>
                )}
              {candidate?.file_type && (
                <Badge variant="outline">{candidate.file_type}</Badge>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            {candidate && (
              <Button
                variant="ghost"
                size="icon-sm"
                title={liked ? t('cvInspector.unlike') : t('cvInspector.like')}
                onClick={() => candidate && like(candidate.id)}
              >
                <Star
                  className={cn(
                    'size-4',
                    liked
                      ? 'fill-amber-400 text-amber-500'
                      : 'text-muted-foreground'
                  )}
                />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <XIcon className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        {candidate && hasText ? (
          <div className="flex min-h-0 flex-1 flex-col gap-3 md:flex-row">
            <div className="shrink-0 overflow-y-auto pb-2 md:w-60 md:border-e md:pe-3">
              <MatchLensControls
                criteria={detail.criteria}
                targetKind={targetKind}
                selectedCriterionId={criterionId}
                onSelectTarget={selectTarget}
                chunkSize={sizeDraft}
                topN={topN}
                onChunkSize={setSizeDraft}
                onTopN={setTopN}
              />
            </div>

            <div className="flex min-h-0 flex-1 flex-col gap-2">
              {(runtimeStatus === 'Starting' ||
                (runtimeStatus && runtimeStatus !== 'Running')) && (
                <div className="flex shrink-0 items-center gap-2 rounded-md border bg-muted/40 px-2 py-1 text-xs text-muted-foreground">
                  <span>
                    {runtimeStatus === 'Starting'
                      ? t('cvInspector.engineStarting')
                      : t('cvInspector.engineNotReady')}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    className="ms-auto h-6 px-2 text-xs"
                    onClick={() => {
                      retryRuntime()
                      chunkQuery.refetch()
                    }}
                  >
                    {t('cvInspector.retry')}
                  </Button>
                </div>
              )}
              {chunkQuery.error && (
                <div className="flex shrink-0 items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1 text-xs text-destructive">
                  <span className="min-w-0 truncate">
                    {t('cvInspector.matchFailed', {
                      message: errorMessage(chunkQuery.error),
                    })}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    className="ms-auto h-6 shrink-0 px-2 text-xs"
                    onClick={() => chunkQuery.refetch()}
                  >
                    {t('cvInspector.retry')}
                  </Button>
                </div>
              )}
              <HighlightedCv
                text={candidate.extracted_text ?? ''}
                result={chunkQuery.data}
                isFetching={chunkQuery.isFetching}
                chunkSize={chunkSize}
                topN={topN}
              />
            </div>
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
              {candidate?.extracted_text ??
                (isLoading ? 'Loading\u2026' : t('cvInspector.noText'))}
            </pre>
          </div>
        )}

        {candidate && !candidate.original_available && (
          <p className="shrink-0 text-xs text-amber-600">
            {t('cvInspector.noOriginal')}
          </p>
        )}

        <div className="flex shrink-0 items-center gap-2 border-t pt-3">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrev}
            disabled={!hasPrev}
          >
            <ChevronLeft /> {t('cvInspector.prev')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleNext}
            disabled={!hasNext}
          >
            {t('cvInspector.next')} <ChevronRight />
          </Button>
          <Button
            size="sm"
            className="ms-auto"
            onClick={() => {
              if (candidate)
                openPath(candidate.source_file_path).catch(e =>
                  toast.error(
                    t('cvInspector.cannotOpen', { message: errorMessage(e) })
                  )
                )
            }}
            disabled={!candidate || !candidate.original_available}
          >
            <ExternalLink /> {t('cvInspector.openOriginal')}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
