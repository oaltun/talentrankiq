import { useLayoutEffect, useRef, useState } from 'react'

import { Spinner } from '@/components/ui/spinner'
import {
  buildLens,
  buildSegments,
  chipLabel,
  formatChunkTooltip,
  formatSpanTooltip,
  layoutChips,
  rankPool,
} from '@/lib/match-lens'
import type { ChipGroup, LensSpan } from '@/lib/match-lens'
import type { ChunkMatchResult } from '@/lib/tauri-bindings'

/** Same-line tolerance (px) for gutter chip collisions. Two anchors on one
 * line share offsetTop; adjacent lines differ by the full line height. */
const CHIP_COLLISION_TOLERANCE_PX = 8

interface HighlightedCvProps {
  text: string
  result: ChunkMatchResult | undefined
  isFetching: boolean
  chunkSize: number
  topN: number
}

/**
 * The CV text pane of the Match Lens: scrollable extracted text with the top-n
 * chunk matches highlighted in yellow and zero-reflow rank chips in the left
 * gutter. Chips hide behind a better-ranked chip sharing their line (the
 * loser's rank folds into the winner's tooltip).
 */
export function HighlightedCv({
  text,
  result,
  isFetching,
  chunkSize,
  topN,
}: HighlightedCvProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const anchorRefs = useRef(new Map<number, HTMLElement>())
  const [tops, setTops] = useState<Record<number, number>>({})

  const spans = result ? buildLens(result.chunks, topN) : []
  // One chip per SELECTED chunk (not per merged paint span): every rank the
  // user asked for gets a number, even when its chunk merged into a neighbor.
  const chips = result ? rankPool(result.chunks, topN) : []
  const segments = buildSegments(
    text.length,
    spans,
    chips.map(c => c.start)
  )
  const anchorsAt = new Map<number, typeof chips>()
  for (const chunk of chips) {
    const list = anchorsAt.get(chunk.start) ?? []
    list.push(chunk)
    anchorsAt.set(chunk.start, list)
  }

  // Re-run measurement when the anchor set or positions change (new data, n,
  // or text); ResizeObserver below covers width-driven reflows.
  const anchorSig = `${chips.map(c => `${c.rank}:${c.start}`).join(',')}@${text.length}`

  useLayoutEffect(() => {
    const container = containerRef.current
    if (!container) return
    const measure = () => {
      const next: Record<number, number> = {}
      for (const [rank, el] of anchorRefs.current) next[rank] = el.offsetTop
      setTops(prev => {
        const keys = Object.keys(next)
        if (keys.length !== Object.keys(prev).length) return next
        for (const k of keys) {
          const n = Number(k)
          if (Math.abs((prev[n] ?? NaN) - (next[n] ?? NaN)) > 0.5) return next
        }
        return prev
      })
    }
    measure()
    const ro =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => measure())
    if (ro) ro.observe(container)
    return () => ro?.disconnect()
  }, [anchorSig])

  // Show every chip whose anchor has been measured; skip unmeasured ones
  // instead of hiding ALL chips until the full set is in. Same-line chips
  // combine into one label — no rank number is ever hidden.
  const measured = chips.filter(c => tops[c.rank] != null)
  const groups =
    measured.length > 0
      ? layoutChips(
          measured,
          chunk => tops[chunk.rank] ?? 0,
          CHIP_COLLISION_TOLERANCE_PX
        )
      : []

  const tooltipFor = (span: LensSpan): string =>
    formatSpanTooltip(span, chunkSize)

  const chipTitle = (group: ChipGroup): string =>
    group.ranks
      .map((rank, i) => formatChunkTooltip(rank, chunkSize, group.sims[i] ?? 0))
      .join('\n')

  return (
    <div className="relative min-h-0 flex-1 overflow-y-auto">
      {isFetching && (
        <Spinner className="absolute end-2 top-2 z-20 size-4 text-muted-foreground" />
      )}
      <div
        ref={containerRef}
        className="relative ps-[2.5ch] font-sans text-sm leading-relaxed break-words whitespace-pre-wrap"
      >
        {segments.map((seg, i) => {
          const anchorChunks = anchorsAt.get(seg.start)
          return (
            <span key={i}>
              {anchorChunks?.map(anchorChunk => (
                <span
                  key={anchorChunk.rank}
                  ref={el => {
                    if (el) anchorRefs.current.set(anchorChunk.rank, el)
                    else anchorRefs.current.delete(anchorChunk.rank)
                  }}
                  className="inline-block size-0"
                  aria-hidden
                />
              ))}
              {seg.kind === 'text' ? (
                text.slice(seg.start, seg.end)
              ) : (
                <mark
                  title={tooltipFor(seg.span)}
                  className="box-decoration-clone rounded-sm bg-yellow-300/60 text-current dark:bg-yellow-500/40"
                >
                  {text.slice(seg.start, seg.end)}
                </mark>
              )}
            </span>
          )
        })}

        {groups.map(g => (
          <div
            key={g.rank}
            title={chipTitle(g)}
            style={{ top: (tops[g.rank] ?? 0) - 10 }}
            className="absolute start-0 z-10 flex h-5 min-w-5 items-center justify-center rounded-full bg-yellow-400 px-1 text-[10px] font-semibold tabular-nums text-black"
          >
            ({chipLabel(g.ranks)})
          </div>
        ))}
      </div>
    </div>
  )
}
