import { ArrowLeft, Pen, X } from "lucide-react"
import { useState } from "react"
import { useHashRouter } from "../../../lib/hash-router"
import { useI18n } from "../../../i18n"
import type { Job } from "../../../services/jobs"
import { Button } from "../../ui/button"

interface WorkspaceHeaderProps {
  job: Job | null
  isLoading: boolean
  onTitleChange: (title: string) => void
}

export function WorkspaceHeader({
  job,
  isLoading,
  onTitleChange,
}: WorkspaceHeaderProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState("")
  const { navigate } = useHashRouter()
  const { t } = useI18n()

  const handleEditClick = () => {
    if (job) {
      setEditTitle(job.title)
      setIsEditing(true)
    }
  }

  const handleSave = () => {
    if (editTitle.trim()) {
      onTitleChange(editTitle.trim())
    }
    setIsEditing(false)
  }

  const handleCancel = () => {
    setEditTitle(job?.title || "")
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave()
    if (e.key === "Escape") handleCancel()
  }

  if (isLoading) {
    return (
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
    )
  }

  return (
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
      {isEditing ? (
        <div className="flex flex-1 items-center gap-2">
          <input
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleSave}
            autoFocus
            className="flex-1 rounded border bg-background px-3 py-1.5 text-sm shadow-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          />
          <Button
            type="button"
            onClick={handleSave}
            variant="ghost"
            size="icon"
            aria-label={t("workspace.saveTitle")}
          >
            <X className="size-4" />
          </Button>
        </div>
      ) : (
        <>
          <h1 className="flex-1 truncate text-lg font-semibold">
            {job?.title || t("workspace.title")}
          </h1>
          {job && (
            <Button
              type="button"
              onClick={handleEditClick}
              variant="ghost"
              size="icon"
              aria-label={t("workspace.editTitle")}
            >
              <Pen className="size-4" />
            </Button>
          )}
        </>
      )}
    </div>
  )
}
