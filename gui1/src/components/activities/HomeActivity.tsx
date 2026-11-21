import { useState, useEffect } from "react"
import { Plus, Loader2, MoreHorizontal, Archive, Trash2 } from "lucide-react"
import {
  listJobs,
  createJob,
  archiveJob,
  deleteJob,
  type JobSummary,
} from "../../services/jobs"
import { useHashRouter } from "../../lib/hash-router"
import { useI18n } from "../../i18n"
import { Button } from "../ui/button"

function relativeDay(
  iso: string,
  t: (key: string, vars?: Record<string, string | number>) => string
): string {
  const then = new Date(`${iso.replace(" ", "T")}Z`).getTime()
  const days = Math.floor((Date.now() - then) / 86_400_000)
  if (days <= 0) return t("jobs.today")
  return t("jobs.daysAgo", { days })
}

function JobCard({
  job,
  onArchive,
  onDelete,
  onOpen,
}: {
  job: JobSummary
  onArchive: (jobId: string, archived: boolean) => void
  onDelete: (jobId: string) => void
  onOpen: () => void
}) {
  const { t } = useI18n()
  const {
    id,
    title,
    cv_imported,
    cv_failed,
    archived,
    last_ranked_at,
    importing,
  } = job

  return (
    <div
      data-testid={`job-item-${id}`}
      onClick={onOpen}
      className="job-card flex cursor-pointer items-center gap-3 rounded-md border bg-card p-4 transition-colors hover:bg-accent/40"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="job-title truncate font-medium">{title}</span>
          {archived && (
            <span className="inline-flex items-center rounded bg-primary/15 text-xs text-primary">
              {t("jobs.archived")}
            </span>
          )}
          {importing && (
            <Loader2 className="size-3 animate-spin text-muted-foreground" />
          )}
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          <span className="inline-flex items-center rounded border px-1.5 py-0.5">
            {t("jobs.cvs", { count: cv_imported })}
          </span>
          {cv_failed > 0 && (
            <span className="inline-flex items-center rounded border px-1.5 py-0.5">
              {t("jobs.failed", { count: cv_failed })}
            </span>
          )}
          {last_ranked_at && (
            <span>
              {t("jobs.ranked", { when: relativeDay(last_ranked_at, t) })}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1">
        <Button
          type="button"
          aria-label={archived ? t("jobs.restore") : t("jobs.archive")}
          title={archived ? t("jobs.restore") : t("jobs.archive")}
          onClick={(e) => {
            e.stopPropagation()
            onArchive(id, !archived)
          }}
          variant="ghost"
          size="icon"
        >
          <Archive className="size-4" />
        </Button>
        <Button
          type="button"
          aria-label={t("jobs.delete")}
          title={t("jobs.delete")}
          onClick={(e) => {
            e.stopPropagation()
            onDelete(id)
          }}
          variant="ghost"
          size="icon"
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="size-4" />
        </Button>
        <Button
          type="button"
          aria-label={t("jobs.more")}
          title={t("jobs.more")}
          onClick={(e) => e.stopPropagation()}
          variant="ghost"
          size="icon"
        >
          <MoreHorizontal className="size-4" />
        </Button>
      </div>
    </div>
  )
}

export function HomeActivity() {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { navigate } = useHashRouter()
  const { t } = useI18n()

  const loadJobs = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const data = await listJobs()
      setJobs(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : t("jobs.error.load"))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadJobs()
  }, [])

  const handleNewJob = async () => {
    try {
      setError(null)
      await createJob()
      await loadJobs()
    } catch (e) {
      setError(e instanceof Error ? e.message : t("jobs.error.create"))
    }
  }

  const handleArchive = async (jobId: string, archived: boolean) => {
    try {
      await archiveJob(jobId, archived)
      await loadJobs()
    } catch (e) {
      setError(e instanceof Error ? e.message : t("jobs.error.update"))
    }
  }

  const handleDelete = async (jobId: string) => {
    try {
      await deleteJob(jobId)
      await loadJobs()
    } catch (e) {
      setError(e instanceof Error ? e.message : t("jobs.error.delete"))
    }
  }

  const active = jobs.filter((j) => !j.archived)

  return (
    <>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">{t("jobs.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("jobs.subtitle")}</p>
        </div>
        <Button
          type="button"
          data-testid="add-job-button"
          aria-label={t("jobs.addJob")}
          title={t("jobs.addJob")}
          onClick={handleNewJob}
          variant="default"
          size="sm"
        >
          <Plus className="size-4" />
          {t("jobs.addJob")}
        </Button>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          {t("jobs.loading")}
        </div>
      )}

      {error && <div className="text-sm text-destructive">{error}</div>}

      {!isLoading && !error && jobs.length === 0 && (
        <div className="job-empty flex flex-col items-center justify-center gap-3 rounded-md border border-dashed py-20 text-center">
          <p className="text-muted-foreground">{t("jobs.noJobs")}</p>
          <Button
            type="button"
            data-testid="add-job-button-empty"
            aria-label={t("jobs.addJob")}
            title={t("jobs.addJob")}
            onClick={handleNewJob}
            variant="default"
            size="sm"
          >
            <Plus className="size-4" />
            {t("jobs.addJob")}
          </Button>
        </div>
      )}

      {!isLoading && !error && active.length > 0 && (
        <div data-testid="job-list" className="job-list flex flex-col gap-3">
          {active.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              onArchive={handleArchive}
              onDelete={handleDelete}
              onOpen={() => navigate(`/job/${job.id}`)}
            />
          ))}
        </div>
      )}
    </>
  )
}
