import { Folder, AlertCircle, ExternalLink, Pencil, Loader2 } from "lucide-react"
import { useState, useEffect } from "react"
import {
  setJobCvDir,
  getJobDetail,
  type Job,
  type JobDetail,
} from "../../../services/jobs"
import { open } from "@tauri-apps/plugin-dialog"
import { AdFileCard } from "./AdFileCard"
import { Button } from "../../ui/button"
import { processInWaves } from "../../../services/rank"
import { listenRankProgress, listenScoutProgress, listenHeavyProgress } from "../../../services/progress"
import type { RankProgress, EmbeddingProgress } from "../../../types"
import { Input } from "../../ui/input"
import { useI18n } from "../../../i18n"

interface SetupPaneProps {
  job: Job | null
  jobDetail: JobDetail | null
  onJobUpdate: (job: Job) => void
  onJobDetailUpdate?: (detail: JobDetail) => void
}

export function SetupPane({
  job,
  jobDetail,
  onJobUpdate,
  onJobDetailUpdate,
}: SetupPaneProps) {
  const { t } = useI18n()
  const [cvDirInput, setCvDirInput] = useState(() => {
    const rawPath = job?.import_directory_path || ""
    return rawPath.startsWith("/") ? rawPath : "/" + rawPath
  })
  const [cvDirError, setCvDirError] = useState<string | null>(null)
  const [rankProgress, setRankProgress] = useState<RankProgress | null>(null)
  const [quickProgress, setQuickProgress] = useState<EmbeddingProgress | null>(null)
  const [deepProgress, setDeepProgress] = useState<EmbeddingProgress | null>(null)
  const jobId = job?.id


  useEffect(() => {
    const rawPath = job?.import_directory_path || ""
    setCvDirInput(rawPath)
  }, [job?.import_directory_path])

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    const offRank = listenRankProgress((p) => {
      if (!cancelled) setRankProgress(p)
    })
    const offScout = listenScoutProgress((p) => {
      if (!cancelled) setQuickProgress(p)
    })
    const offHeavy = listenHeavyProgress((p) => {
      if (!cancelled) setDeepProgress(p)
    })
    return () => {
      cancelled = true
      void offRank.then((f) => f()).catch(() => {})
      void offScout.then((f) => f()).catch(() => {})
      void offHeavy.then((f) => f()).catch(() => {})
    }
  }, [jobId])

  const commitCvDir = async (rawPath: string) => {
    if (!jobId) return
    const path = rawPath.trim()
    if (!path) return
    const normalizedPath = path.startsWith("/") ? path : "/" + path

    const current = (job?.import_directory_path || "").trim()
    const normalizedCurrent = current.startsWith("/")
      ? current
      : current
        ? "/" + current
        : ""
    if (normalizedPath === normalizedCurrent) {
      return
    }

    setCvDirInput(normalizedPath)
    setCvDirError(null)
    try {
      const updated = await setJobCvDir(jobId, normalizedPath)
      onJobUpdate(updated)
      const detail = await getJobDetail(jobId)
      onJobDetailUpdate?.(detail)
      await processInWaves(jobId)
    } catch (e: unknown) {
      const msg =
        e instanceof Error
          ? e.message
          : typeof e === "string"
            ? e
            : "Failed to change CV directory"
      setCvDirError(msg)
    }
  }

  const handleChangeCvDir = async () => {
    if (!jobId) return
    setCvDirError(null)
    try {
      const selected = await open({ directory: true, multiple: false })
      const path = Array.isArray(selected)
        ? (selected[0] ?? null)
        : typeof selected === "string"
          ? selected
          : null
      if (path) {
        await commitCvDir(path)
      }
    } catch (e: unknown) {
      const msg =
        e instanceof Error
          ? e.message
          : typeof e === "string"
            ? e
            : "Failed to change CV directory"
      setCvDirError(msg)
    }
  }

  const handleCvDirBlur = () => {
    commitCvDir(cvDirInput)
  }

  const handleCvDirKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.currentTarget.blur()
    }
  }

  const handleOpenCvDir = async () => {
    if (job?.import_directory_path) {
      await import("@tauri-apps/plugin-opener").then(({ openPath }) =>
        openPath(job.import_directory_path!)
      )
    }
  }

  return (
    <div className="space-y-4">
      <AdFileCard jobId={jobId!} />

      <div className="rounded-md border bg-card p-3">
        <div
          className="mb-2 flex items-center justify-between"
          data-testid="cv-directory-header"
        >
          <div className="flex items-center gap-2">
            <Folder className="size-5 text-muted-foreground" />
            <span className="font-medium" data-testid="cv-directory-label">
              {t("workspace.cvWatchDirectory")}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              onClick={handleChangeCvDir}
              data-testid="change-cv-dir-btn"
              variant="outline"
              size="sm"
              aria-label={t("workspace.changeDirectory")}
            >
              <Pencil className="size-4" />
            </Button>

            {job?.import_directory_path && (
              <Button
                type="button"
                onClick={handleOpenCvDir}
                variant="outline"
                size="sm"
                data-testid="open-cv-dir-btn"
                title={t("workspace.openInFileManager")}
              >
                <ExternalLink className="size-4" />
              </Button>
            )}
          </div>
        </div>

        <div className="mb-2 flex items-center gap-2">
          <Input
            type="text"
            value={cvDirInput}
            onChange={(e) => setCvDirInput(e.target.value)}
            onBlur={handleCvDirBlur}
            onKeyDown={handleCvDirKeyDown}
            placeholder={t("workspace.enterCvDirPath")}
            className="min-w-0 flex-1 text-left"
            data-testid="cv-dir-input"
          />
        </div>

        {cvDirError && (
          <div className="flex items-center gap-2 text-sm text-destructive">
            <AlertCircle className="size-4" />
            <span>{cvDirError}</span>
          </div>
        )}
      </div>

      {(rankProgress?.running ||
        deepProgress?.done ||
        quickProgress?.done ||
        (rankProgress?.compared ?? 0) > 0) && (
        <div
          className="rounded-md border bg-card p-3 text-sm"
          data-testid="progress-card"
        >
          <div className="mb-2 flex items-center gap-2">
            {rankProgress?.running && (
              <Loader2
                className="size-4 animate-spin text-muted-foreground"
                data-testid="progress-spinner"
              />
            )}
            <span className="font-medium" data-testid="progress-title">
              {t("workspace.progress")}
            </span>
          </div>
          <div className="space-y-1">
            <div
              className="flex items-center justify-between"
              data-testid="progress-quick-done"
            >
              <span className="text-muted-foreground">
                {t("workspace.progressQuickEmbedded")}
              </span>
              <span className="tabular-nums" data-testid="progress-quick-count">
                {quickProgress?.done ?? 0}
              </span>
            </div>
            <div
              className="flex items-center justify-between"
              data-testid="progress-deep-done"
            >
              <span className="text-muted-foreground">
                {t("workspace.progressDeepEmbedded")}
              </span>
              <span className="tabular-nums" data-testid="progress-deep-count">
                {deepProgress?.done ?? 0}
              </span>
              {(deepProgress?.done ?? 0) > 0 && (
                <span className="sr-only" data-testid="progress-deep-ready">
                  ready
                </span>
              )}
            </div>
            <div
              className="flex items-center justify-between"
              data-testid="progress-compared-done"
            >
              <span className="text-muted-foreground">
                {t("workspace.progressCompared")}
              </span>
              <span className="tabular-nums" data-testid="progress-compared-count">
                {rankProgress?.compared ?? 0}
              </span>
            </div>
            {jobDetail?.job?.job_ad_text?.trim() &&
              (jobDetail?.cv_quick_similarity_count ?? 0) > 0 && (
                <div
                  className="flex items-center justify-between"
                  data-testid="progress-quick-similarity-done"
                >
                  <span className="text-muted-foreground">
                    {t("workspace.progressQuickSimilarity")}
                  </span>
                  <span
                    className="tabular-nums"
                    data-testid="progress-quick-similarity-count"
                  >
                    {jobDetail?.cv_quick_similarity_count ?? 0}
                  </span>
                </div>
              )}
            {jobDetail?.job?.job_ad_text?.trim() &&
              (jobDetail?.cv_deep_similarity_count ?? 0) > 0 && (
                <div
                  className="flex items-center justify-between"
                  data-testid="progress-deep-similarity-done"
                >
                  <span className="text-muted-foreground">
                    {t("workspace.progressDeepSimilarity")}
                  </span>
                  <span
                    className="tabular-nums"
                    data-testid="progress-deep-similarity-count"
                  >
                    {jobDetail?.cv_deep_similarity_count ?? 0}
                  </span>
                </div>
              )}
          </div>
        </div>
      )}

      {jobDetail && jobDetail.criteria.length > 0 && (
        <div className="rounded-md border bg-card p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-medium">
                {t("workspace.additionalCriteria")}
              </span>
              <span className="inline-flex items-center rounded border px-1.5 py-0.5 text-xs text-muted-foreground">
                {jobDetail.criteria.length}
              </span>
            </div>
          </div>
          <div className="space-y-1">
            {jobDetail.criteria.map((criterion) => (
              <div
                key={criterion.id}
                className="flex items-center gap-2 text-sm"
              >
                <span className="flex-1 truncate">{criterion.text}</span>
                <span className="inline-flex items-center rounded bg-primary/10 px-2 py-0.5 text-xs text-primary">
                  {t("workspace.weight")}: {criterion.weight}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

{/*       {jobDetail && jobDetail.rubric.length > 0 && (
        <div className="rounded-md border bg-card p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-medium">{t("workspace.rubric")}</span>
            <span className="inline-flex items-center rounded border px-1.5 py-0.5 text-xs text-muted-foreground">
              {t("workspace.capacity")}:{" "}
              {Math.round(jobDetail.rubric_capacity || 0)}
            </span>
          </div>
          <div className="space-y-1">
            {jobDetail.rubric.map((q) => (
              <div
                key={q.question_id}
                className="flex items-center gap-2 text-sm"
              >
                <span className="flex-1 truncate">
                  {translateRubricLabel(q.label)}
                </span>
                <span className="inline-flex items-center rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  {q.max_points} {t("workspace.points")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )} */}
    </div>
  )
}
