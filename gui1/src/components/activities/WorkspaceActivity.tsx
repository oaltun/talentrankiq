import { Loader2, ArrowLeft, Folder, AlertCircle } from "lucide-react"
import { useState, useEffect, useCallback } from "react"
import {
  watchJob,
  getJobDetail,
  updateJob,
  type Job,
  type JobDetail,
} from "../../services/jobs"
import type { CandidateDetail } from "../../types"
import { useHashRouter } from "../../lib/hash-router"
import {
  getScoutModel,
  listEmbedders,
} from "../../services/models"
import {
  rememberWorkspaceJob,
  getLastWorkspaceJob,
} from "../../lib/last-workspace-job"
import {
  WorkspaceHeader,
  SetupPane,
  CandidatesPane,
  RubricDialog,
  ExportView,
  CvInspectorDialog,
} from "./workspace"
import { InfoBar } from "../layout"
import { Button } from "../ui/button"
import { useI18n } from "../../i18n"

export function WorkspaceActivity({ jobId }: { jobId: string | null }) {
  const [job, setJob] = useState<Job | null>(null)
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showRubricDialog, setShowRubricDialog] = useState(false)
  const [showExportView, setShowExportView] = useState(false)
  const [selectedCandidate, setSelectedCandidate] =
    useState<CandidateDetail | null>(null)
  const [candidateNeighbors, setCandidateNeighbors] = useState<{
    prev_id?: string | null
    next_id?: string | null
  } | null>(null)
  const [showComparatorInfobar, setShowComparatorInfobar] = useState(false)
  const [jobNotFound, setJobNotFound] = useState(false)
  const { navigate } = useHashRouter()
  const { t } = useI18n()

  const loadJob = useCallback(async () => {
    if (!jobId) {
      setJob(null)
      setJobDetail(null)
      setError(null)
      setJobNotFound(false)
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      setJobNotFound(false)
      const [jobData, detailData] = await Promise.all([
        watchJob(jobId),
        getJobDetail(jobId),
      ])
      setJob(jobData)
      setJobDetail(detailData)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      if (msg.toLowerCase().includes("not found")) {
        setJobNotFound(true)
        setError(null)
      } else {
        setError(msg)
      }
    } finally {
      setIsLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    loadJob()
  }, [loadJob])

  useEffect(() => {
    let cancelled = false
    async function checkModels() {
      try {
        const quick = await getScoutModel()
        if (cancelled) return
        if (quick && !quick.downloaded) {
          // Quick model not downloaded; job setup doesn't need it.
          // Ranking will be disabled until model is downloaded.
        }
        const embedders = await listEmbedders()
        if (cancelled) return
        const defaultEmbedder = embedders.find((e) => e.is_default) ?? null
        if (!defaultEmbedder || !defaultEmbedder.downloaded) {
          // Deep model not downloaded; job setup doesn't need it.
          // Ranking will be disabled until model is downloaded.
        }
        const comparator = defaultEmbedder?.comparator ?? null
        if (comparator && comparator.entitled && !comparator.exists) {
          setShowComparatorInfobar(true)
        }
      } catch {
        // ignore in environments without CLI
      }
    }
    checkModels()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (jobId) {
      rememberWorkspaceJob(jobId)
    } else {
      const lastJobId = getLastWorkspaceJob()
      if (lastJobId) navigate(`/job/${lastJobId}`)
    }
  }, [jobId, navigate])

  useEffect(() => {
    if (jobNotFound) {
      const timer = setTimeout(() => {
        navigate("/models")
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [jobNotFound, navigate])

  const handleTitleChange = async (title: string) => {
    if (!job) return
    try {
      const updated = await updateJob(
        job.id,
        title,
        job.job_ad_text ?? null,
        job.triqui_size
      )
      setJob(updated)
    } catch (e) {
      setError(e instanceof Error ? e.message : t("workspace.errorLoadingJob"))
    }
  }

  const handleJobUpdate = (updatedJob: Job) => {
    setJob(updatedJob)
  }

  const handleCloseCandidate = () => {
    setSelectedCandidate(null)
    setCandidateNeighbors(null)
  }

  const handleNavigateCandidate = (direction: "prev" | "next") => {
    if (!candidateNeighbors) return
    const nextId =
      direction === "prev"
        ? candidateNeighbors.prev_id
        : candidateNeighbors.next_id
    if (!nextId || !jobDetail) return
    const candidate = jobDetail.triqui.find((c) => c.id === nextId)
    if (candidate) {
      // We need to fetch the full candidate detail
      // This is handled by the CvInspectorDialog itself
    }
  }

  if (!jobId) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-md border border-dashed py-20 text-center">
        <Folder className="size-12 text-muted-foreground/50" />
        <h1 className="text-lg font-semibold">{t("workspace.title")}</h1>
        <p className="text-sm text-muted-foreground">
          {t("workspace.selectJobToStart")}
        </p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <>
        <div className="mb-4 flex items-center gap-2">
          <Button
            type="button"
            aria-label={t("workspace.backToJobs")}
            title={t("workspace.backToJobs")}
            onClick={() => navigate("/jobs")}
            variant="ghost"
            size="icon"
          >
            <ArrowLeft className="size-4" />
          </Button>
          <h1 className="text-lg font-semibold">{t("workspace.title")}</h1>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          {t("workspace.loadingJob")}
        </div>
      </>
    )
  }

  if (jobNotFound) {
    return (
      <>
        <div className="mb-4 flex items-center gap-2">
          <h1 className="text-lg font-semibold">{t("workspace.title")}</h1>
        </div>
        <div className="flex flex-col items-center justify-center gap-4 rounded-md border p-8 text-center">
          <AlertCircle className="size-12 text-destructive" />
          <div className="text-lg font-medium">
            {t("workspace.jobNotFound")}
          </div>
          <p className="text-sm text-muted-foreground">
            {t("workspace.jobNotFoundDescription")}
          </p>
          <p className="text-xs text-muted-foreground">
            {t("workspace.redirectingToJobs")}
          </p>
        </div>
      </>
    )
  }

  if (error) {
    return (
      <>
        <div className="mb-4 flex items-center gap-2">
          <Button
            type="button"
            aria-label={t("workspace.backToJobs")}
            title={t("workspace.backToJobs")}
            onClick={() => navigate("/jobs")}
            variant="ghost"
            size="icon"
          >
            <ArrowLeft className="size-4" />
          </Button>
          <h1 className="text-lg font-semibold">{t("workspace.title")}</h1>
        </div>
        <div className="text-sm text-destructive">{error}</div>
      </>
    )
  }

  return (
    <>
      <WorkspaceHeader
        job={job}
        isLoading={isLoading}
        onTitleChange={handleTitleChange}
      />

      {showComparatorInfobar && (
        <InfoBar
          isVisible={true}
          variant="warning"
          title={t("workspace.comparatorMissingTitle")}
          description={t("workspace.comparatorMissingDescription")}
          actionLabel={t("workspace.getComparator")}
          onAction={() => navigate("/models?tab=deep")}
        />
      )}

      {job && jobDetail && (
        <div
          className="grid min-h-[600px] grid-cols-1 gap-4 lg:grid-cols-3"
          data-testid="workspace-page"
          data-job-id={job.id}
        >
          {/* Left column - Setup */}
          <div className="lg:col-span-1" data-testid="workspace-left-pane">
            <SetupPane
              job={job}
              jobDetail={jobDetail}
              onJobUpdate={handleJobUpdate}
              onJobDetailUpdate={setJobDetail}
            />
          </div>

          {/* Right column - Candidates */}
          <div className="lg:col-span-2" data-testid="workspace-right-pane">
            <CandidatesPane jobId={job.id} jobDetail={jobDetail} />
          </div>
        </div>
      )}

      <RubricDialog
        open={showRubricDialog}
        job={job}
        onClose={() => setShowRubricDialog(false)}
        onUpdateWeights={async (weights) => {
          if (!job) return
          try {
            const { updateJobWeights } = await import("../../services/jobs")
            const updated = await updateJobWeights(
              job.id,
              weights.adWeight,
              weights.fbJobWeight,
              weights.fbProfileWeight
            )
            setJob(updated)
          } catch (e) {
            setError(
              e instanceof Error ? e.message : t("workspace.errorLoadingJob")
            )
          }
        }}
      />

      <ExportView
        open={showExportView}
        jobId={jobId!}
        jobDetail={jobDetail}
        onClose={() => setShowExportView(false)}
      />

      {selectedCandidate && (
        <CvInspectorDialog
          open={true}
          candidate={selectedCandidate}
          neighbors={candidateNeighbors}
          jobDetail={jobDetail}
          onClose={handleCloseCandidate}
          onNavigate={handleNavigateCandidate}
        />
      )}
    </>
  )
}
