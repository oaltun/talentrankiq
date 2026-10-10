/**
 * Pure logic for the Match Lens highlight module of the CV Inspector dialog.
 *
 * The backend returns EVERY chunk (char offsets + similarity) at the chosen
 * size; this module does everything the frontend still has to do:
 *   1. pick the top-n chunks,
 *   2. number them 1..n by similarity (rank chips),
 *   3. merge overlapping chunks into a single paint span,
 *   4. flatten the spans into plain text segments,
 *   5. lay out gutter rank chips, hiding same-line collisions behind the
 *      better-ranked chip (the loser's number folds into the winner's tooltip).
 */

/** A chunk as returned by the backend (`ChunkSim` in the rpc contract). */
export interface LensChunk {
  char_start: number
  char_end: number
  similarity: number
}

/** A selected chunk with its rank. */
export interface RankedChunk {
  start: number
  end: number
  similarity: number
  rank: number
}

/** One paintable highlight span after overlap merging. */
export interface LensSpan {
  start: number
  end: number
  /** Member chunk ranks, best (lowest) first; merged spans list several. */
  ranks: number[]
  /** Member similarities, aligned with `ranks`. */
  sims: number[]
  bestRank: number
}

export type LensSegment =
  | { kind: 'text'; start: number; end: number }
  | { kind: 'span'; start: number; end: number; span: LensSpan }

/** Chunk-size bounds enforced server-side; mirrored here to pre-clamp inputs. */
export const MIN_CHUNK_TOKENS = 8
export const MAX_CHUNK_TOKENS = 2048

/** Clamp a user-entered chunk size to `MIN..MAX` (integer). */
export function clampChunkSize(size: number): number {
  return Math.min(
    Math.max(Math.round(size) || MIN_CHUNK_TOKENS, MIN_CHUNK_TOKENS),
    MAX_CHUNK_TOKENS
  )
}

/** Radio-picker sentinel id for the job ad (first entry of the target picker). */
export const JOB_AD_TARGET_ID = '__jobad__'

/** The top-n chunks by similarity (document order breaks ties). */
export function topN(chunks: LensChunk[], n: number): LensChunk[] {
  if (n <= 0 || chunks.length === 0) return []
  return chunks
    .map((chunk, index) => ({ chunk, index }))
    .sort(
      (a, b) => b.chunk.similarity - a.chunk.similarity || a.index - b.index
    )
    .slice(0, n)
    .map(({ chunk }) => chunk)
}

/** Select the top-n and rank them 1..n by similarity descending. */
export function rankPool(chunks: LensChunk[], n: number): RankedChunk[] {
  return topN(chunks, n)
    .map(c => ({
      start: c.char_start,
      end: c.char_end,
      similarity: c.similarity,
    }))
    .sort(
      (a, b) =>
        b.similarity - a.similarity || a.start - b.start || a.end - b.end
    )
    .map((c, i) => ({ ...c, rank: i + 1 }))
}

/**
 * Merge overlapping (or touching) chunks into single spans. Merged spans keep
 * every member rank + sim, ordered best rank first, and paint with the best
 * member's chip.
 */
export function mergeRanks(chunks: RankedChunk[]): LensSpan[] {
  const sorted = [...chunks].sort((a, b) => a.start - b.start || a.end - b.end)
  const spans: LensSpan[] = []
  let current: RankedChunk[] = []
  const flush = () => {
    if (current.length === 0) return
    const members = [...current].sort((a, b) => a.rank - b.rank)
    const best = members[0]
    if (!best) return
    spans.push({
      start: Math.min(...members.map(m => m.start)),
      end: Math.max(...members.map(m => m.end)),
      ranks: members.map(m => m.rank),
      sims: members.map(m => m.similarity),
      bestRank: best.rank,
    })
    current = []
  }
  for (const c of sorted) {
    if (current.length > 0 && c.start <= Math.max(...current.map(m => m.end))) {
      current.push(c)
    } else {
      flush()
      current = [c]
    }
  }
  flush()
  return spans
}

/** Select + rank + merge, in paint order (by document position). */
export function buildLens(chunks: LensChunk[], n: number): LensSpan[] {
  return mergeRanks(rankPool(chunks, n)).sort((a, b) => a.start - b.start)
}

/** Flatten text into plain/highlighted segments. `chipStarts` (per-chunk
 * anchors) are split too: a chunk that merged into a neighbor's paint span
 * still gets a segment boundary at its own start, so its gutter anchor can
 * attach at the right line. */
export function buildSegments(
  textLength: number,
  spans: LensSpan[],
  chipStarts: number[] = []
): LensSegment[] {
  const usable: LensSpan[] = spans
    .map(s => ({
      ...s,
      start: Math.max(0, Math.min(s.start, textLength)),
      end: Math.max(0, Math.min(s.end, textLength)),
    }))
    .filter(s => s.end > s.start)
  if (usable.length === 0)
    return textLength > 0 ? [{ kind: 'text', start: 0, end: textLength }] : []

  const boundaries = new Set<number>([0, textLength])
  for (const s of usable) {
    boundaries.add(s.start)
    boundaries.add(s.end)
  }
  for (const p of chipStarts) {
    if (p >= 0 && p <= textLength) boundaries.add(p)
  }
  const points = [...boundaries].sort((a, b) => a - b)
  const segments: LensSegment[] = []
  for (let i = 0; i < points.length - 1; i++) {
    const start = points[i]
    const end = points[i + 1]
    if (start == null || end == null || start === end) continue
    const covering = usable.filter(s => s.start <= start && end <= s.end)
    if (covering.length === 0) {
      segments.push({ kind: 'text', start, end })
      continue
    }
    const winner = covering.reduce((a, b) => (b.bestRank < a.bestRank ? b : a))
    segments.push({ kind: 'span', start, end, span: winner })
  }
  return segments
}

/** One gutter chip: every selected chunk gets one, at its own chunk start.
 * Chips whose anchors land on the same line are COMBINED into a single chip
 * (joined ranks label) so no rank number is ever hidden. */
export interface ChipGroup {
  /** Best (lowest) rank in the group — owns the anchor line. */
  rank: number
  /** Gutter top in px of the group's anchor line. */
  top: number
  /** All member ranks, best first. */
  ranks: number[]
  /** Member similarities, aligned with `ranks`. */
  sims: number[]
}

/**
 * Group per-chunk chips by anchor line: chunks whose anchors are within
 * `tolerance` px share one combined chip. `topFor(chunk)` must return the
 * gutter top px for the chunk's anchor.
 */
export function layoutChips(
  chunks: RankedChunk[],
  topFor: (chunk: RankedChunk) => number,
  tolerance: number
): ChipGroup[] {
  const ordered = [...chunks].sort((a, b) => a.rank - b.rank)
  const groups: ChipGroup[] = []
  for (const c of ordered) {
    const top = topFor(c)
    const group = groups.find(g => Math.abs(g.top - top) < tolerance)
    if (group) {
      group.ranks.push(c.rank)
      group.sims.push(c.similarity)
    } else {
      groups.push({ rank: c.rank, top, ranks: [c.rank], sims: [c.similarity] })
    }
  }
  return groups.sort((a, b) => a.rank - b.rank)
}

/** Chip label for a group: joined ranks (`3·4`); collapses to `3+4` (best
 * rank + count of the rest) when the joined label would overflow the gutter. */
export function chipLabel(ranks: number[]): string {
  const best = ranks[0] ?? 0
  const joined = ranks.join('·')
  return joined.length > 8 ? `${best}+${ranks.length - 1}` : joined
}

/** One tooltip line for a chunk member, e.g. `#3 (256t) · sim 0.71`. */
export function formatChunkTooltip(
  rank: number,
  sizeTokens: number,
  sim: number
): string {
  return `#${rank} (${sizeTokens}t) · sim ${sim.toFixed(2)}`
}

/** Full tooltip for a span: one line per member, best first. */
export function formatSpanTooltip(span: LensSpan, sizeTokens: number): string {
  return span.ranks
    .map((rank, i) => formatChunkTooltip(rank, sizeTokens, span.sims[i] ?? 0))
    .join('\n')
}
