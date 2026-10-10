import { useTranslation } from 'react-i18next'

import { cn } from '@/lib/utils'
import { MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS } from '@/lib/match-lens'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { Criterion } from '@/lib/tauri-bindings'

export type LensTargetKind = 'jobAd' | 'criterion'

/** Picker labels render the starting ~40 chars + `…` when longer. */
function pickerLabel(s: string): string {
  const oneLine = s.replace(/\s+/g, ' ').trim()
  return oneLine.length > 40 ? `${oneLine.slice(0, 40).trimEnd()}…` : oneLine
}

interface MatchLensControlsProps {
  criteria: Criterion[]
  targetKind: LensTargetKind
  selectedCriterionId: string | null
  onSelectTarget: (kind: LensTargetKind, criterionId: string | null) => void
  chunkSize: number
  topN: number
  onChunkSize: (v: number) => void
  onTopN: (v: number) => void
}

export function MatchLensControls({
  criteria,
  targetKind,
  selectedCriterionId,
  onSelectTarget,
  chunkSize,
  topN,
  onChunkSize,
  onTopN,
}: MatchLensControlsProps) {
  const { t } = useTranslation()

  return (
    <div className="flex flex-col gap-3">
      <div
        className="flex flex-col gap-1"
        role="radiogroup"
        aria-label={t('cvInspector.pickTarget')}
      >
        <TargetOption
          active={targetKind === 'jobAd'}
          onSelect={() => onSelectTarget('jobAd', null)}
          label={t('cvInspector.target.jobAd')}
        />
        {criteria.map(c => (
          <TargetOption
            key={c.id}
            active={targetKind === 'criterion' && selectedCriterionId === c.id}
            onSelect={() => onSelectTarget('criterion', c.id)}
            label={pickerLabel(c.text)}
          />
        ))}
      </div>

      <div className="flex flex-col gap-2 border-t pt-3">
        <div className="flex items-center gap-1">
          <Label
            htmlFor="match-lens-size"
            className="shrink-0 text-xs text-muted-foreground"
          >
            {t('cvInspector.chunkSize')}
          </Label>
          <Input
            id="match-lens-size"
            type="number"
            min={MIN_CHUNK_TOKENS}
            max={MAX_CHUNK_TOKENS}
            value={chunkSize}
            onChange={e => onChunkSize(Number(e.target.value))}
            className="h-7"
          />
          <span className="text-xs whitespace-nowrap text-muted-foreground">
            {t('cvInspector.tokens')}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-sm">
          <span className="text-muted-foreground">{t('cvInspector.top')}</span>
          <Input
            id="match-lens-topn"
            type="number"
            min={1}
            max={50}
            value={topN}
            onChange={e =>
              onTopN(
                Math.min(
                  Math.max(Math.round(Number(e.target.value)) || 1, 1),
                  50
                )
              )
            }
            className="h-7 w-16"
          />
          <span className="text-muted-foreground">
            {t('cvInspector.chunks')}
          </span>
        </div>
      </div>
    </div>
  )
}

function TargetOption({
  active,
  onSelect,
  label,
}: {
  active: boolean
  onSelect: () => void
  label: string
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      className={cn(
        'flex items-center gap-2 rounded-md px-2 py-1 text-start text-sm transition-colors',
        active ? 'bg-accent font-medium' : 'hover:bg-muted'
      )}
      onClick={onSelect}
    >
      <span
        className={cn(
          'size-2 shrink-0 rounded-full border',
          active ? 'border-primary bg-primary' : 'border-muted-foreground'
        )}
      />
      <span className="truncate">{label}</span>
    </button>
  )
}
