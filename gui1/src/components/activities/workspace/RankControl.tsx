import { Loader2, AlertCircle, CheckCircle, RotateCcw, HelpCircle } from "lucide-react"
import { useState, useEffect, useCallback, useRef } from "react"
import {
  getJobUiState,
  getRankState,
  processInWaves,
} from "../../../services/rank"
import { listenRankProgress } from "../../../services/progress"
import type { JobUiState, RankState, RankProgress } from "../../../types"
import { Button } from "../../ui/button"
import { useI18n } from "../../../i18n"

interface RankControlProps {
  jobId: string
  onRunFinished: () => void
}

export function RankControl({ jobId, onRunFinished }: RankControlProps) {
  const { t } = useI18n()
  const [jobUiState, setJobUiState] = useState<JobUiState | null>(null)
  const [rankState, setRankState] = useState<RankState | null>(null)
  const [isPolling, setIsPolling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sequenceRef = useRef(0)
  const isSubscribedRef = useRef(false)

  const fetchJobUiState = useCallback(async () => {
    try {
      const state = await getJobUiState(jobId)
      setJobUiState(state)
      return state
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch job UI state"
      setError(msg)
      return null
    }
  }, [jobId])

  const fetchRankState = useCallback(async () => {
    try {
      const state = await getRankState(jobId)
      setRankState(state)
      return state
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to fetch rank state"
      setError(msg)
      return null
    }
  }, [jobId])

  const startPolling = useCallback(async () => {
    if (isPolling) return
    setIsPolling(true)
    sequenceRef.current += 1
    const currentSequence = sequenceRef.current

    while (isPolling && sequenceRef.current === currentSequence) {
      await fetchJobUiState()
      const currentRank = rankState ?? (await fetchRankState())
      if (!currentRank?.running) {
        setIsPolling(false)
        break
      }
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }, [fetchJobUiState, fetchRankState, isPolling, rankState])

  const stopPolling = useCallback(() => {
    setIsPolling(false)
    sequenceRef.current += 1
  }, [])

  useEffect(() => {
    fetchJobUiState()
    fetchRankState()
  }, [fetchJobUiState, fetchRankState])

  useEffect(() => {
    if (isSubscribedRef.current) return
    isSubscribedRef.current = true

    const subscribe = async () => {
      const off = await listenRankProgress((progress: RankProgress) => {
        if (progress.job_id !== jobId) return
        if (progress.running) {
          setRankState((prev) => prev ? { ...prev, running: true, stage: progress.stage, wave: progress.wave } : null)
          startPolling()
        } else if (progress.stage === "done") {
          stopPolling()
          fetchJobUiState()
          fetchRankState()
          onRunFinished()
        }
      })
      return off
    }

    const offPromise = subscribe()
    return () => {
      offPromise.then((off) => off?.())
      isSubscribedRef.current = false
    }
  }, [jobId, startPolling, stopPolling, fetchJobUiState, fetchRankState, onRunFinished])

  const deriveStatus = (): { label: string; variant: "running" | "stale" | "converged" | "blocked" | "cancelled" | "partial"; reasonKey?: string } => {
    if (!jobUiState && !rankState) {
      return { label: t("rank.status.running"), variant: "running" }
    }

    if (rankState?.running) {
      return { label: t("rank.status.running"), variant: "running" }
    }

    const uiReason = jobUiState?.reason
    const rankReason = rankState?.reason

    if (uiReason?.startsWith("blocked:")) {
      const code = uiReason.replace("blocked:", "")
      return { label: t(`rank.reason.${code}`), variant: "blocked", reasonKey: code }
    }
    if (rankReason?.startsWith("blocked:")) {
      const code = rankReason.replace("blocked:", "")
      return { label: t(`rank.reason.${code}`), variant: "blocked", reasonKey: code }
    }

    if (rankReason === "cancelled") {
      return { label: t("rank.status.cancelled"), variant: "cancelled" }
    }

    if (rankState?.converged) {
      return { label: t("rank.status.converged"), variant: "converged" }
    }

    if (rankReason === "stale" || uiReason === "stale") {
      return { label: t("rank.status.stale"), variant: "stale" }
    }

    if (rankReason === "partial" || rankReason === "clean") {
      return { label: t(`rank.status.${rankReason}`), variant: "partial" }
    }

    return { label: t("rank.status.stale"), variant: "stale" }
  }

  const handleReRank = async () => {
    if (!jobUiState?.sort_allowed || rankState?.running) return
    try {
      await processInWaves(jobId)
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to start ranking"
      setError(msg)
    }
  }

  const status = deriveStatus()
  const isRunning = rankState?.running ?? false
  const canRerank = jobUiState?.sort_allowed === true && !isRunning
  const rerankReason = jobUiState?.reason ?? rankState?.reason ?? "stale"

  return (
    <div className="rounded-md border bg-card p-3" data-testid="rank-control">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-medium" data-testid="rank-control-title">
          {t("workspace.rankStatus")}
        </span>
        {error && (
          <span className="text-sm text-destructive" data-testid="rank-control-error">
            {error}
          </span>
        )}
      </div>

      <div className="mb-2 flex items-center gap-2">
        {status.variant === "running" && (
          <Loader2 className="size-4 animate-spin text-primary" data-testid="rank-status-spinner" />
        )}
        {status.variant === "converged" && (
          <CheckCircle className="size-4 text-green-500" data-testid="rank-status-converged" />
        )}
        {status.variant === "blocked" && (
          <AlertCircle className="size-4 text-destructive" data-testid="rank-status-blocked" />
        )}
        {status.variant === "cancelled" && (
          <HelpCircle className="size-4 text-muted-foreground" data-testid="rank-status-cancelled" />
        )}
        {status.variant === "stale" && (
          <RotateCcw className="size-4 text-muted-foreground" data-testid="rank-status-stale" />
        )}
        {status.variant === "partial" && (
          <RotateCcw className="size-4 text-muted-foreground" data-testid="rank-status-partial" />
        )}
        <span
          className="font-medium"
          data-testid="rank-status-label"
          data-reason={status.reasonKey ?? ""}
        >
          {status.label}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Button
          type="button"
          onClick={handleReRank}
          disabled={!canRerank}
          variant={canRerank ? "default" : "outline"}
          size="sm"
          data-testid="rerank-button"
          data-reason={rerankReason}
        >
          <RotateCcw className="size-4 mr-1" />
          {t("workspace.rerank")}
        </Button>
        {!canRerank && rerankReason && (
          <span className="text-sm text-muted-foreground" data-testid="rerank-disabled-reason">
            {t(`rank.reason.${rerankReason.replace("blocked:", "")}`)}
          </span>
        )}
      </div>
    </div>
  )
}