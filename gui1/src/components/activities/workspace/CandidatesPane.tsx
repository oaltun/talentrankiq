import {
  Search,
  ChevronUp,
  ChevronDown,
  Star,
  Eye,
  Loader2,
  Filter,
  X,
  Users,
} from "lucide-react"
import { useState, useEffect, useCallback } from "react"
import {
  listCandidates,
  getCandidate,
  candidateNeighbors,
  applyFeedback,
  resetFeedback,
  listLikedCandidateIds,
  type CandidatePage,
  type CandidateSummary,
  type CandidateDetail,
  type CandidateNeighbors,
} from "../../../services/candidates"
import { useFeedbackStore } from "../../../lib/stores/feedback-store"
import type { JobDetail } from "../../../types"
import { Button } from "../../ui/button"
import { useI18n } from "../../../i18n"

interface CandidatesPaneProps {
  jobId: string
  jobDetail: JobDetail | null
}

const SORT_OPTIONS = [
  { key: "__rank", label: "Rank" },
  { key: "__name", label: "Name" },
  { key: "__total", label: "Total" },
  { key: "__space", label: "Space" },
  { key: "__source", label: "Source" },
] as const

type SortKey = (typeof SORT_OPTIONS)[number]["key"] | (string & {})

interface SortConfig {
  key: SortKey
  dir: "asc" | "desc"
}

export function CandidatesPane({ jobId, jobDetail }: CandidatesPaneProps) {
  const { t } = useI18n()
  const [candidates, setCandidates] = useState<CandidateSummary[]>([])
  const [total, setTotal] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortConfig, setSortConfig] = useState<SortConfig>({
    key: "__rank",
    dir: "asc",
  })
  const [page, setPage] = useState(0)
  const [selectedCandidate, setSelectedCandidate] =
    useState<CandidateDetail | null>(null)
  const [neighbors, setNeighbors] = useState<CandidateNeighbors | null>(null)
  const [viewMode, setViewMode] = useState<"table" | "card">("table")
  const [error, setError] = useState<string | null>(null)
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false)

  const likedIds = useFeedbackStore((state) => state.likedIdsByJob[jobId]) ?? []

  const pageSize = jobDetail?.job.triqui_size ?? 10
  const totalPages = Math.ceil(total / pageSize)

  const loadCandidates = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)
      const result: CandidatePage = await listCandidates(
        jobId,
        page * pageSize,
        pageSize,
        searchQuery || undefined,
        sortConfig.key,
        sortConfig.dir === "asc"
      )
      setCandidates(result.items)
      setTotal(result.total)
    } catch (e) {
      const message =
        e instanceof Error ? e.message : "Failed to load candidates"
      if (
        message.toLowerCase().includes("not found") ||
        message.toLowerCase().includes("no such job") ||
        message.toLowerCase().includes("no candidates") ||
        message === "Failed to load candidates"
      ) {
        setCandidates([])
        setTotal(0)
        setError(null)
      } else {
        setError(message)
      }
    } finally {
      setIsLoading(false)
      setHasLoadedOnce(true)
    }
  }, [jobId, page, pageSize, searchQuery, sortConfig])

  useEffect(() => {
    loadCandidates()
  }, [loadCandidates])

  // The CV import and ranking run in the background after a directory
  // change; keep refreshing so imported/ranked candidates appear without
  // the user having to switch away and back. Poll continuously but only
  // update state when the rows or their rank/points change.
  useEffect(() => {
    let lastSignature = JSON.stringify(
      candidates.map((c) => [c.id, c.rank, c.points])
    )
    const interval = setInterval(async () => {
      try {
        const result = await listCandidates(
          jobId,
          page * pageSize,
          pageSize,
          searchQuery || undefined,
          sortConfig.key,
          sortConfig.dir === "asc"
        )
        const nextSignature = JSON.stringify(
          result.items.map((c) => [c.id, c.rank, c.points])
        )
        if (nextSignature !== lastSignature || result.total !== total) {
          lastSignature = nextSignature
          setCandidates(result.items)
          setTotal(result.total)
        }
      } catch {
        // Ignore polling errors
      }
    }, 2500)
    return () => clearInterval(interval)
  }, [jobId, page, pageSize, searchQuery, sortConfig, total, candidates])

  const handleSort = (key: SortKey) => {
    setSortConfig((prev) => ({
      key,
      dir: prev.key === key && prev.dir === "asc" ? "desc" : "asc",
    }))
    setPage(0)
  }

  const handleOpenCandidate = async (candidate: CandidateSummary) => {
    try {
      const detail = await getCandidate(candidate.id)
      setSelectedCandidate(detail)
      const nav = await candidateNeighbors(jobId, candidate.id)
      setNeighbors(nav)
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to load candidate detail"
      )
    }
  }

  const handleCloseCandidate = () => {
    setSelectedCandidate(null)
    setNeighbors(null)
  }

  const handleNavigateCandidate = async (direction: "prev" | "next") => {
    if (!neighbors) return
    const nextId = direction === "prev" ? neighbors.prev_id : neighbors.next_id
    if (!nextId) return
    const candidate = candidates.find((c) => c.id === nextId)
    if (candidate) {
      await handleOpenCandidate(candidate)
    }
  }

  const handleToggleLike = async (candidateId: string) => {
    const newLikedIds = likedIds.includes(candidateId)
      ? likedIds.filter((id) => id !== candidateId)
      : [...likedIds, candidateId]

    try {
      await applyFeedback(
        jobId,
        candidates.map((c) => c.id),
        newLikedIds
      )
      useFeedbackStore.getState().setLikedIds(jobId, newLikedIds)
      // Refresh candidates to get updated liked status
      loadCandidates()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update feedback")
    }
  }

  const handleResetFeedback = async () => {
    try {
      await resetFeedback(jobId)
      useFeedbackStore.getState().setLikedIds(jobId, [])
      loadCandidates()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to reset feedback")
    }
  }

  const loadLikedIds = async () => {
    try {
      const ids = await listLikedCandidateIds(jobId)
      useFeedbackStore.getState().setLikedIds(jobId, ids)
    } catch (e) {
      console.error("Failed to load liked IDs:", e)
    }
  }

  useEffect(() => {
    loadLikedIds()
  }, [jobId])

  const getSortIcon = (key: SortKey) => {
    if (sortConfig.key !== key) return null
    return sortConfig.dir === "asc" ? (
      <ChevronUp className="size-4" />
    ) : (
      <ChevronDown className="size-4" />
    )
  }

  if (isLoading && !hasLoadedOnce) {
    return (
      <div
        className="flex h-64 items-center justify-center"
        data-testid="candidates-pane"
      >
        <Loader2
          className="size-8 animate-spin text-muted-foreground"
          data-testid="candidates-spinner"
        />
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="flex h-64 items-center justify-center text-destructive"
        data-testid="candidates-pane"
      >
        <span>{error}</span>
      </div>
    )
  }

  if (candidates.length === 0 && total === 0) {
    return (
      <div
        className="flex h-64 flex-col items-center justify-center gap-3 text-center"
        data-testid="candidates-pane"
      >
        <p className="text-muted-foreground" data-testid="candidates-empty">
          {t("workspace.noCandidatesFound")}
        </p>
        <p className="text-sm text-muted-foreground">
          {t("workspace.importCvsToStart")}
        </p>
      </div>
    )
  }

  return (
    <div
      className="flex h-full flex-col rounded-md border bg-card p-3"
      data-testid="candidates-pane"
    >
      <div className="mb-4 flex items-center gap-2">
        <Users className="size-5 text-muted-foreground" />
        <span className="font-medium" data-testid="candidates-label">
          {t("workspace.candidates")}
        </span>
      </div>
      {/* Toolbar */}
      <div className="mb-4 flex items-center gap-2">
        <div className="relative max-w-xs flex-1">
          <Search className="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder={t("workspace.searchCandidates")}
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              setPage(0)
            }}
            className="w-full rounded border bg-background py-1.5 pr-3 pl-9 text-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          />
          {searchQuery && (
            <Button
              type="button"
              onClick={() => setSearchQuery("")}
              variant="ghost"
              size="icon"
              className="absolute top-1/2 right-3 -translate-y-1/2"
            >
              <X className="size-4" />
            </Button>
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button
            type="button"
            onClick={() => setViewMode("table")}
            variant={viewMode === "table" ? "default" : "ghost"}
            size="icon"
            aria-label={t("workspace.tableView")}
          >
            <Filter className="size-4" />
          </Button>
          <Button
            type="button"
            onClick={() => setViewMode("card")}
            variant={viewMode === "card" ? "default" : "ghost"}
            size="icon"
            aria-label={t("workspace.cardView")}
          >
            <Eye className="size-4" />
          </Button>
        </div>

        {likedIds.length > 0 && (
          <Button
            type="button"
            onClick={handleResetFeedback}
            variant="destructive"
            size="sm"
          >
            <X className="size-4" />
            {t("workspace.clearFeedback")}
          </Button>
        )}
      </div>

      {/* Results Table */}
      <div className="flex-1 overflow-auto">
        <table
          className="w-full caption-bottom text-sm"
          data-testid="candidates-pane"
        >
          <thead className="[&_tr]:border-b">
            <tr className="border-b bg-muted/50">
              <th
                className="cursor-pointer px-3 py-2 text-left hover:bg-accent/50"
                onClick={() => handleSort("__rank")}
              >
                {t("workspace.rank")} {getSortIcon("__rank")}
              </th>
              <th
                className="cursor-pointer px-3 py-2 text-left hover:bg-accent/50"
                onClick={() => handleSort("__name")}
              >
                {t("workspace.candidate")} {getSortIcon("__name")}
              </th>
              <th
                className="cursor-pointer px-3 py-2 text-right hover:bg-accent/50"
                onClick={() => handleSort("__total")}
              >
                {t("workspace.total")} {getSortIcon("__total")}
              </th>
              <th className="px-3 py-2 text-center">Like</th>
              <th className="px-3 py-2 text-center">View</th>
            </tr>
          </thead>
          <tbody>{candidates.map((candidate) => (
          <tr
            key={candidate.id}
            className="cursor-pointer border-t hover:bg-accent/50"
            onClick={() => handleOpenCandidate(candidate)}
          >
            <td className="px-3 py-2 text-center font-mono text-sm">{candidate.rank ?? "—"}</td>
            <td className="px-3 py-2">
              <div className="flex items-center gap-2">
                <span className="truncate font-medium">{candidate.file_name}</span>
                <span className="inline-flex items-center rounded border px-1.5 py-0.5 text-xs text-muted-foreground">
                  {candidate.file_type}
                </span>
                <span className="inline-flex items-center rounded border px-1.5 py-0.5 text-xs text-muted-foreground">
                  {candidate.extraction_status}
                </span>
              </div>
            </td>
            <td className="px-3 py-2 text-right">
              {candidate.points !== null && candidate.points !== undefined ? (
                <span className="font-mono">{candidate.points.toFixed(1)}</span>
              ) : (
                <span className="text-muted-foreground">—</span>
              )}
            </td>
            <td className="px-3 py-2 text-center">
              <Button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  handleToggleLike(candidate.id)
                }}
                variant="ghost"
                size="icon"
                aria-label={likedIds.includes(candidate.id) ? "Unlike" : "Like"}
              >
                <Star
                  className={`size-4 ${likedIds.includes(candidate.id) ? "fill-current text-yellow-500" : "text-muted-foreground"}`}
                />
              </Button>
            </td>
            <td className="px-3 py-2 text-center">
              <Button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  handleOpenCandidate(candidate)
                }}
                variant="ghost"
                size="icon"
                aria-label="View details"
              >
                <Eye className="size-4" />
              </Button>
            </td>
          </tr>
        ))}</tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-4 flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {t("workspace.pageOf", { page: page + 1, totalPages, total })}
          </span>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              variant="outline"
              size="sm"
            >
              {t("workspace.previous")}
            </Button>
            <Button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              variant="outline"
              size="sm"
            >
              {t("workspace.next")}
            </Button>
          </div>
        </div>
      )}

      {/* Candidate Inspector Overlay */}
      {selectedCandidate && (
        <CandidateInspector
          candidate={selectedCandidate}
          neighbors={neighbors}
          onClose={handleCloseCandidate}
          onNavigate={handleNavigateCandidate}
        />
      )}
    </div>
  )
}

function CandidateInspector({
  candidate,
  neighbors,
  onClose,
  onNavigate,
}: {
  candidate: CandidateDetail
  neighbors: CandidateNeighbors | null
  onClose: () => void
  onNavigate: (direction: "prev" | "next") => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <div>
            <h2 className="text-lg font-semibold">{candidate.file_name}</h2>
            <p className="text-sm text-muted-foreground">
              {candidate.file_type} • {candidate.extraction_status}
              {candidate.rank && (
                <span className="ml-2">Rank: {candidate.rank}</span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {neighbors?.prev_id && (
              <Button
                type="button"
                onClick={() => onNavigate("prev")}
                variant="outline"
                size="sm"
              >
                Previous
              </Button>
            )}
            {neighbors?.next_id && (
              <Button
                type="button"
                onClick={() => onNavigate("next")}
                variant="outline"
                size="sm"
              >
                Next
              </Button>
            )}
            <Button type="button" onClick={onClose} variant="ghost" size="icon">
              <X className="size-5" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 space-y-4 overflow-auto p-4">
          {/* Scores */}
          {candidate.points !== undefined && candidate.points !== null && (
            <div className="rounded border bg-muted p-4">
              <h3 className="mb-2 font-medium">Scores</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Total Points</span>
                  <p className="text-2xl font-bold">
                    {candidate.points?.toFixed(1) ?? "—"}
                  </p>
                </div>
                <div>
                  <span className="text-muted-foreground">Max Points</span>
                  <p className="text-2xl font-bold">
                    {candidate.points_max?.toFixed(1) ?? "—"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Rubric Breakdown */}
          {candidate.rubric_scores && candidate.rubric_scores.length > 0 && (
            <div className="rounded border bg-muted p-4">
              <h3 className="mb-2 font-medium">Rubric Breakdown</h3>
              <div className="space-y-2">
                {candidate.rubric_scores.map((score) => (
                  <div
                    key={score.question_id}
                    className="flex items-center justify-between"
                  >
                    <span className="text-sm">{score.label}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono">
                        {score.points.toFixed(1)} /{" "}
                        {score.max_points.toFixed(1)}
                      </span>
                      <div className="h-2 w-32 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full bg-primary"
                          style={{
                            width: `${(score.points / score.max_points) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Extracted Text */}
          {candidate.extracted_text && (
            <div className="rounded border bg-muted p-4">
              <h3 className="mb-2 font-medium">Extracted Text</h3>
              <pre className="max-h-96 overflow-auto font-mono text-sm whitespace-pre-wrap text-muted-foreground">
                {candidate.extracted_text}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
