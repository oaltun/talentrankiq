import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from '@tanstack/react-router'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useJob, useUpdateJob } from '@/services/jobs'
import { useUIStore } from '@/store/ui-store'
import { commands } from '@/lib/tauri-bindings'
import { errorMessage } from '@/lib/errors'
import { logger } from '@/lib/logger'
import { SetupPane } from './SetupPane'
import { RubricDialog } from './RubricDialog'
import { CandidatesPane } from './CandidatesPane'
import { ExportView } from './ExportView'
import { CvInspectorDialog } from './CvInspectorDialog'
import { ChevronLeft, Loader2, Pen } from 'lucide-react'

/**
 * Workspace (`/jobs/:jobId`): Setup (left) | Candidates (right).
 *
 * Candidate details appear in an overlay dialog when a row is clicked. The
 * overlay's prev/next buttons walk every CV (via `candidateNeighbors`), honoring
 * the search. The Setup pane's "Scoring & rubric" card opens the job's rubric
 * dialog (all max-points weights) — the only entry point, so the modal has
 * exactly one door.
 *
 * The job ad lives in the Setup pane, not in a pane of its own: it is a file the
 * user owns, its controls are two buttons, and a third column bought it a
 * permanent share of the width that the candidate table needs.
 */
export function JobView() {
  const { jobId } = useParams({ strict: false })
  const navigate = useNavigate()
  const { data, isLoading } = useJob(jobId)
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(
    null
  )
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState(false)
  // Export drawer over the workspace (Setup pane stays visible behind it).
  const [showExport, setShowExport] = useState(false)
  // Scoring & rubric dialog (weights), opened from the Setup pane's card.
  const [rubricOpen, setRubricOpen] = useState(false)
  const updateJob = useUpdateJob(jobId ?? '')
  const [editingTitle, setEditingTitle] = useState(false)
  const [titleDraft, setTitleDraft] = useState(data?.job.title ?? '')
  const titleInputRef = useRef<HTMLInputElement>(null)

  // Remember the open job so the "Workspace" activity-bar item can return to it.
  useEffect(() => {
    if (jobId) useUIStore.getState().setCurrentJobId(jobId)
  }, [jobId])

  // Only the currently viewed job's CV directory is watched (single-watcher
  // policy): switching jobs restarts watching and fires a catch-up auto-import
  // scan, so CVs dropped here while another job was open are imported too.
  useEffect(() => {
    if (!jobId) return
    commands
      .watchJobCvDir(jobId)
      .catch(e =>
        logger.warn('cv dir watch failed', { jobId, error: errorMessage(e) })
      )
  }, [jobId])

  if (isLoading || !data || !jobId) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        <Loader2 className="size-5 animate-spin" />
      </div>
    )
  }

  const job = data.job
  if (!editingTitle && titleDraft !== job.title) setTitleDraft(job.title)

  const commitTitle = () => {
    setEditingTitle(false)
    const t = titleDraft.trim()
    if (!t || t === job.title) {
      setTitleDraft(job.title)
      return
    }
    updateJob.mutate(
      {
        title: t,
        jobAdText: job.job_ad_text ?? null,
        triquiSize: job.triqui_size,
      },
      {
        onSuccess: () => toast.success('Job renamed'),
        onError: e => {
          setTitleDraft(job.title)
          toast.error(`Could not rename job: ${errorMessage(e)}`)
        },
      }
    )
  }

  const handleCloseCandidate = () => setSelectedCandidateId(null)

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => navigate({ to: '/' })}
          title="Back to Jobs"
        >
          <ChevronLeft className="size-4" />
        </Button>
        {editingTitle ? (
          <Input
            ref={titleInputRef}
            autoFocus
            value={titleDraft}
            onChange={e => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={e => {
              if (e.key === 'Enter') commitTitle()
              if (e.key === 'Escape') {
                setTitleDraft(job.title)
                setEditingTitle(false)
              }
            }}
            className="h-7 max-w-96 text-sm"
          />
        ) : (
          <button
            type="button"
            className="group flex min-w-0 items-center gap-1"
            title="Click to edit title"
            onClick={() => {
              setTitleDraft(job.title)
              setEditingTitle(true)
            }}
          >
            <span className="truncate text-sm font-medium">{job.title}</span>
            <Pen className="size-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
          </button>
        )}
        <div className="ml-auto" />
      </header>

      <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4 md:flex-row md:overflow-hidden">
        <SetupPane
          jobId={jobId}
          detail={data}
          onOpenRubric={() => setRubricOpen(true)}
        />

        <CandidatesPane
          jobId={jobId}
          detail={data}
          selectedId={selectedCandidateId}
          onSelect={setSelectedCandidateId}
          onExport={() => setShowExport(true)}
          query={query}
          onQueryChange={setQuery}
          sortBy={sortBy}
          onSortByChange={setSortBy}
          sortDir={sortDir}
          onSortDirChange={setSortDir}
        />

        <RubricDialog
          key={jobId}
          jobId={jobId}
          detail={data}
          open={rubricOpen}
          onOpenChange={setRubricOpen}
        />

        <ExportView
          jobId={jobId}
          detail={data}
          open={showExport}
          onOpenChange={setShowExport}
        />

        <CvInspectorDialog
          jobId={jobId}
          detail={data}
          candidateId={selectedCandidateId}
          query={query}
          sortBy={sortBy}
          sortDir={sortDir}
          onSelect={setSelectedCandidateId}
          onClose={handleCloseCandidate}
        />
      </div>
    </div>
  )
}
