import { FileText, ExternalLink, AlertCircle, Pencil, Loader2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { getJobAd, setJobAdPath, type JobAdFile } from "../../../services/jobs"
import { open } from "@tauri-apps/plugin-dialog"
import { openPath } from "@tauri-apps/plugin-opener"
import { Button } from "../../ui/button"
import { useI18n } from "../../../i18n"
import { Input } from "../../ui/input"

interface AdFileCardProps {
  jobId: string
}

export function AdFileCard({ jobId }: AdFileCardProps) {
  const { t } = useI18n()
  const [adFile, setAdFile] = useState<JobAdFile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [adPathInput, setAdPathInput] = useState("")

  const loadAdFile = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      const file = await getJobAd(jobId)
      setAdFile(file)
      setAdPathInput(file.path ?? "")
    } catch (e) {
      setError(e instanceof Error ? e.message : t("workspace.failedToLoadJobAd"))
    } finally {
      setIsLoading(false)
    }
  }, [jobId, t])

  const commitAdPath = async (rawPath: string) => {
    const value = rawPath ?? ""
    const path = value.trim()
    if (!path) return
    const file = await setJobAdPath(jobId, path)
    setAdFile(file)
    setAdPathInput(file.path ?? path)
  }

  const handleChangePath = async () => {
    try {
      setError(null)
      const result = await open({
        directory: false,
        multiple: false,
        filters: [{ name: "Text Files", extensions: ["txt"] }],
      })
      const filePath =
        Array.isArray(result) ? String(result[0] ?? "") : typeof result === "string" ? result : ""
      if (filePath) {
        const file = await setJobAdPath(jobId, filePath)
        setAdFile(file)
        setAdPathInput(file.path ?? filePath)
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg || t("workspace.failedToChangeJobAdPath"))
    }
  }

  const handleOpen = async () => {
    if (adFile?.path) {
      try {
        await openPath(adFile.path)
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        setError(msg || t("workspace.failedToOpenFileInEditor"))
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.currentTarget.blur()
    }
  }

  useEffect(() => {
    void loadAdFile()
  }, [jobId, loadAdFile])

  if (isLoading) {
    return (
      <div
        className="flex items-center gap-2 rounded-md border bg-card p-3"
        data-testid="ad-file-card"
      >
        <Loader2 className="size-4 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">{t("workspace.loadingJobAd")}</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-md border bg-destructive/10 p-3 text-sm text-destructive">
        <AlertCircle className="size-4" />
        <span>{error}</span>
      </div>
    )
  }

  return (
    <div className="rounded-md border bg-card p-3" data-testid="ad-file-card">
      <div className="mb-2 flex items-center justify-between" data-testid="ad-card-header">
        <div className="flex items-center gap-2">
          <FileText className="size-5 text-muted-foreground" />
          <span className="font-medium" data-testid="job-ad-label">
            {t("workspace.jobAd")}
          </span>
          {adFile?.created && (
            <span className="inline-flex items-center rounded bg-primary/15 text-xs text-primary">
              {t("workspace.autoCreated")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            onClick={() => void handleChangePath()}
            data-testid="change-ad-btn"
            variant="outline"
            size="sm"
            aria-label={t("workspace.changeAd")}
          >
            <Pencil className="size-4" />
          </Button>
          {adFile?.path && (
            <Button
              type="button"
              onClick={() => void handleOpen()}
              variant="outline"
              size="sm"
              data-testid="open-ad-btn"
              title={t("workspace.openInEditor")}
            >
              <ExternalLink className="size-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="mb-2 flex items-center gap-2 text-sm">
        <Input
          type="text"
          value={adPathInput}
          onChange={(e) => setAdPathInput(e.target.value)}
          onBlur={() => void commitAdPath(adPathInput)}
          onKeyDown={handleKeyDown}
          placeholder={t("workspace.enterJobAdPath")}
          className="min-w-0 flex-1 text-left"
          data-testid="ad-path-input"
        />
      </div>
    </div>
  )
}