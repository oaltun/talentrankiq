import { X, ChevronLeft, ChevronRight, Loader2 } from "lucide-react"
import { useState, useEffect } from "react"
import {
  chunkMatch,
  type ChunkMatchResult,
  type ChunkMatchTarget,
  type CandidateDetail,
} from "../../../services/match-lens"
import { MatchLensControls } from "./match-lens/MatchLensControls"
import { HighlightedCv } from "./match-lens/HighlightedCv"
import { Button } from "../../ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../../ui/dialog"

interface CvInspectorDialogProps {
  open: boolean
  candidate: CandidateDetail | null
  neighbors: { prev_id?: string | null; next_id?: string | null } | null
  jobDetail: any
  onClose: () => void
  onNavigate: (direction: "prev" | "next") => void
}

export function CvInspectorDialog({
  open,
  candidate,
  neighbors,
  jobDetail,
  onClose,
  onNavigate,
}: CvInspectorDialogProps) {
  if (!open || !candidate) return null

  const [matchResult, setMatchResult] = useState<ChunkMatchResult | null>(null)
  const [isMatching, setIsMatching] = useState(false)
  const [matchTarget, setMatchTarget] = useState<ChunkMatchTarget>("JobAd")
  const [chunkSize, setChunkSize] = useState(256)

  const availableTargets: { id: ChunkMatchTarget; label: string }[] = [
    { id: "JobAd", label: "Job Ad" },
    ...(jobDetail?.criteria.map((c: any) => ({
      id: { Criterion: { id: c.id } } as ChunkMatchTarget,
      label: c.text,
    })) ?? []),
  ]

  const runMatch = async () => {
    if (!candidate) return
    try {
      setIsMatching(true)
      const result = await chunkMatch(candidate.id, matchTarget, chunkSize)
      setMatchResult(result)
    } catch (e) {
      console.error("Match failed:", e)
    } finally {
      setIsMatching(false)
    }
  }

  useEffect(() => {
    runMatch()
  }, [candidate?.id, matchTarget, chunkSize])

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] max-w-5xl">
        <DialogHeader>
          <DialogTitle>{candidate.file_name}</DialogTitle>
          <DialogDescription>
            {candidate.file_type} • {candidate.extraction_status}
            {candidate.rank && (
              <span className="ml-2">Rank: {candidate.rank}</span>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="flex h-full flex-col">
          {/* Header actions */}
          <div className="flex items-center justify-between border-b p-4">
            <div className="flex items-center gap-2">
              {neighbors?.prev_id && (
                <Button
                  type="button"
                  onClick={() => onNavigate("prev")}
                  variant="outline"
                  size="sm"
                >
                  <ChevronLeft className="mr-1 size-4" />
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
                  <ChevronRight className="ml-1 size-4" />
                </Button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                onClick={onClose}
                variant="ghost"
                size="icon"
              >
                <X className="size-5" />
              </Button>
            </div>
          </div>

          {/* Content */}
          <div className="flex flex-1 overflow-auto">
            {/* Left panel - CV Text with highlights */}
            <div className="w-1/2 overflow-auto border-r p-4">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="font-medium">CV Text</h3>
                <div className="flex items-center gap-2">
                  <MatchLensControls
                    targets={availableTargets}
                    selectedTarget={matchTarget}
                    onTargetChange={setMatchTarget}
                    chunkSize={chunkSize}
                    onChunkSizeChange={setChunkSize}
                    isLoading={isMatching}
                    onRunMatch={runMatch}
                  />
                </div>
              </div>

              <HighlightedCv
                text={candidate.extracted_text ?? ""}
                chunks={matchResult?.chunks ?? []}
                model={matchResult?.model ?? ""}
              />
            </div>

            {/* Right panel - Match Lens details */}
            <div className="w-1/2 overflow-auto p-4">
              <h3 className="mb-4 font-medium">Match Lens</h3>

              {isMatching ? (
                <div className="flex h-64 items-center justify-center">
                  <Loader2 className="size-8 animate-spin text-muted-foreground" />
                </div>
              ) : matchResult ? (
                <div className="space-y-4">
                  <div className="rounded border bg-muted p-3">
                    <p className="text-sm text-muted-foreground">
                      Model: {matchResult.model}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Chunks: {matchResult.chunks.length}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Chunk size: {chunkSize} tokens
                    </p>
                  </div>

                  <div className="max-h-96 overflow-auto rounded border bg-muted p-3">
                    <h4 className="mb-2 font-medium">Top Matching Chunks</h4>
                    {matchResult.chunks
                      .slice()
                      .sort((a, b) => b.similarity - a.similarity)
                      .slice(0, 20)
                      .map((chunk, idx) => {
                        const chunkText = candidate.extracted_text
                          ? candidate.extracted_text.slice(
                              chunk.char_start,
                              chunk.char_end
                            )
                          : `Chars ${chunk.char_start}-${chunk.char_end}`
                        return (
                          <div
                            key={idx}
                            className="mb-2 rounded bg-background p-2"
                          >
                            <div className="mb-1 flex items-center justify-between text-xs">
                              <span>Chunk {idx + 1}</span>
                              <span className="font-mono text-primary">
                                {(chunk.similarity * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="max-h-20 overflow-auto text-sm whitespace-pre-wrap">
                              {chunkText}
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>
              ) : (
                <div className="flex h-64 items-center justify-center text-muted-foreground">
                  <p>Select a target and click "Run Match" to analyze.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
