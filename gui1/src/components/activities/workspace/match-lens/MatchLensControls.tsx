import { Target, Loader2, Play } from "lucide-react"
import type { ChunkMatchTarget } from "../../../../types"
import { Button } from "../../../ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../ui/select"
import { Input } from "../../../ui/input"

interface MatchLensControlsProps {
  targets: { id: ChunkMatchTarget; label: string }[]
  selectedTarget: ChunkMatchTarget
  onTargetChange: (target: ChunkMatchTarget) => void
  chunkSize: number
  onChunkSizeChange: (size: number) => void
  isLoading: boolean
  onRunMatch: () => void
}

export function MatchLensControls({
  targets,
  selectedTarget,
  onTargetChange,
  chunkSize,
  onChunkSizeChange,
  isLoading,
  onRunMatch,
}: MatchLensControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="relative">
        <Target className="absolute top-1/2 left-2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Select
          value={JSON.stringify(selectedTarget)}
          onValueChange={(value) => onTargetChange(JSON.parse(value as string))}
        >
          <SelectTrigger className="pr-8 pl-8">
            <SelectValue placeholder={targets[0]?.label ?? ""} />
          </SelectTrigger>
          <SelectContent>
            {targets.map((t) => (
              <SelectItem
                key={JSON.stringify(t.id)}
                value={JSON.stringify(t.id)}
              >
                {t.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs text-muted-foreground">Chunk size:</label>
        <Input
          type="number"
          value={chunkSize}
          onChange={(e) => onChunkSizeChange(parseInt(e.target.value) || 256)}
          min="8"
          max="2048"
          step="8"
          className="w-20"
        />
      </div>

      <Button
        type="button"
        onClick={onRunMatch}
        disabled={isLoading}
        variant="default"
        size="sm"
      >
        {isLoading ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Play className="size-4" />
        )}
        {isLoading ? "Matching..." : "Run Match"}
      </Button>
    </div>
  )
}
