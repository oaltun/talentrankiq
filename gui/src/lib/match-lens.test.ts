import { describe, expect, it } from 'vitest'

import {
  buildLens,
  buildSegments,
  chipLabel,
  clampChunkSize,
  formatSpanTooltip,
  layoutChips,
  mergeRanks,
  rankPool,
  topN,
} from './match-lens'
import type { LensChunk, LensSpan, RankedChunk } from './match-lens'

const chunk = (start: number, end: number, similarity: number): LensChunk => ({
  char_start: start,
  char_end: end,
  similarity,
})

describe('clampChunkSize', () => {
  it('clamps to the server-side bounds and rounds', () => {
    expect(clampChunkSize(256)).toBe(256)
    expect(clampChunkSize(4)).toBe(8)
    expect(clampChunkSize(9999)).toBe(2048)
    expect(clampChunkSize(64.6)).toBe(65)
    expect(clampChunkSize(NaN)).toBe(8)
  })
})

describe('topN', () => {
  it('picks the n highest similarities, breaking ties by document order', () => {
    const chunks = [
      chunk(0, 10, 0.5),
      chunk(10, 20, 0.9),
      chunk(20, 30, 0.9),
      chunk(30, 40, 0.7),
    ]
    expect(topN(chunks, 2)).toEqual([chunk(10, 20, 0.9), chunk(20, 30, 0.9)])
  })

  it('returns empty for n <= 0 or no chunks', () => {
    expect(topN([chunk(0, 1, 1)], 0)).toEqual([])
    expect(topN([], 5)).toEqual([])
  })
})

describe('rankPool', () => {
  it('numbers the top-n 1..n by similarity descending', () => {
    const ranked = rankPool(
      [chunk(0, 10, 0.8), chunk(40, 50, 0.6), chunk(20, 30, 0.9)],
      3
    )
    expect(ranked.map(r => [r.rank, r.similarity])).toEqual([
      [1, 0.9],
      [2, 0.8],
      [3, 0.6],
    ])
  })

  it('caps at n', () => {
    const chunks = [chunk(0, 10, 0.99), chunk(10, 20, 0.98), chunk(20, 30, 0.5)]
    expect(rankPool(chunks, 2)).toHaveLength(2)
  })
})

describe('mergeRanks', () => {
  it('merges overlapping chunks and keeps member ranks/sims', () => {
    const merged = mergeRanks([
      { start: 0, end: 60, similarity: 0.7, rank: 2 },
      { start: 45, end: 100, similarity: 0.9, rank: 1 },
    ])
    expect(merged).toHaveLength(1)
    expect(merged[0]).toMatchObject({ start: 0, end: 100, bestRank: 1 })
    expect(merged[0]?.ranks).toEqual([1, 2])
    expect(merged[0]?.sims).toEqual([0.9, 0.7])
  })

  it('keeps non-overlapping chunks separate', () => {
    const merged = mergeRanks([
      { start: 0, end: 60, similarity: 0.7, rank: 1 },
      { start: 200, end: 260, similarity: 0.5, rank: 2 },
    ])
    expect(merged).toHaveLength(2)
  })

  it('merges chains of overlapping chunks into one span', () => {
    const merged = mergeRanks([
      { start: 0, end: 50, similarity: 0.5, rank: 3 },
      { start: 40, end: 90, similarity: 0.7, rank: 2 },
      { start: 80, end: 130, similarity: 0.9, rank: 1 },
    ])
    expect(merged).toHaveLength(1)
    expect(merged[0]?.ranks).toEqual([1, 2, 3])
    expect(merged[0]?.end).toBe(130)
  })
})

describe('buildLens', () => {
  it('returns disjoint merged spans in document order', () => {
    const spans = buildLens(
      [chunk(100, 160, 0.8), chunk(0, 60, 0.9), chunk(200, 260, 0.7)],
      5
    )
    expect(spans.map(s => [s.start, s.end])).toEqual([
      [0, 60],
      [100, 160],
      [200, 260],
    ])
    expect(spans.map(s => s.bestRank)).toEqual([1, 2, 3])
  })

  it('chain-merges overlapping chunks across the document', () => {
    const spans = buildLens(
      [chunk(100, 160, 0.8), chunk(0, 60, 0.9), chunk(50, 110, 0.7)],
      5
    )
    expect(spans.map(s => [s.start, s.end])).toEqual([[0, 160]])
    expect(spans[0]?.ranks).toEqual([1, 2, 3])
  })
})

describe('buildSegments', () => {
  const span = (s: {
    start: number
    end: number
    bestRank?: number
  }): LensSpan => ({
    ranks: [s.bestRank ?? 1],
    sims: [0.9],
    bestRank: s.bestRank ?? 1,
    start: s.start,
    end: s.end,
  })

  it('emits plain text around a single span', () => {
    const segs = buildSegments(100, [span({ start: 20, end: 40 })])
    expect(segs).toEqual([
      { kind: 'text', start: 0, end: 20 },
      {
        kind: 'span',
        start: 20,
        end: 40,
        span: expect.objectContaining({ bestRank: 1 }),
      },
      { kind: 'text', start: 40, end: 100 },
    ])
  })

  it('splits at every span boundary', () => {
    const segs = buildSegments(100, [
      span({ start: 10, end: 60, bestRank: 2 }),
      span({ start: 30, end: 50, bestRank: 1 }),
    ])
    const kinds = segs.map(s => s.kind)
    expect(kinds).toEqual(['text', 'span', 'span', 'span', 'text'])
    expect(segs[2]).toMatchObject({ start: 30, end: 50 })
  })

  it('clips out-of-bounds spans to the text length', () => {
    const segs = buildSegments(30, [span({ start: 20, end: 99 })])
    expect(segs).toHaveLength(2)
    expect(segs[1]).toMatchObject({ start: 20, end: 30 })
  })

  it('splits at chip starts interior to a merged span', () => {
    // One merged paint span [10, 60]; chunk 2 starts at 30 inside it.
    const segs = buildSegments(100, [span({ start: 10, end: 60 })], [30])
    expect(segs).toEqual([
      { kind: 'text', start: 0, end: 10 },
      { kind: 'span', start: 10, end: 30, span: expect.anything() },
      { kind: 'span', start: 30, end: 60, span: expect.anything() },
      { kind: 'text', start: 60, end: 100 },
    ])
  })

  it('returns one plain segment with no spans', () => {
    expect(buildSegments(10, [])).toEqual([{ kind: 'text', start: 0, end: 10 }])
  })
})

describe('layoutChips', () => {
  const c = (rank: number): RankedChunk => ({
    start: rank * 10,
    end: rank * 10 + 10,
    similarity: 0.5 + rank * 0.1,
    rank,
  })

  it('gives every chunk a rank; nothing is ever hidden', () => {
    const groups = layoutChips([c(1), c(2), c(3)], () => 42, 8)
    expect(groups).toHaveLength(1)
    expect(groups[0]).toMatchObject({ rank: 1, top: 42 })
    expect(groups[0]?.ranks).toEqual([1, 2, 3])
    expect(groups[0]?.sims).toEqual([0.6, 0.7, 0.8])
  })

  it('keeps chips on different lines separate', () => {
    const groups = layoutChips(
      [c(1), c(2)],
      chunk => (chunk.rank === 1 ? 0 : 100),
      8
    )
    expect(groups).toHaveLength(2)
    expect(groups.map(g => g.rank)).toEqual([1, 2])
  })

  it('groups by anchor line, best rank owns the top', () => {
    const groups = layoutChips(
      [c(1), c(2), c(3)],
      chunk => (chunk.rank === 2 ? 100 : 0),
      8
    )
    expect(groups).toHaveLength(2)
    expect(groups[0]).toMatchObject({ rank: 1, top: 0, ranks: [1, 3] })
    expect(groups[1]).toMatchObject({ rank: 2, top: 100, ranks: [2] })
  })
})

describe('chipLabel', () => {
  it('joins few ranks with dots', () => {
    expect(chipLabel([3])).toBe('3')
    expect(chipLabel([3, 4])).toBe('3·4')
  })

  it('collapses wide groups to best+count', () => {
    expect(chipLabel([1, 2, 3, 4, 5, 6])).toBe('1+5')
  })
})

describe('formatSpanTooltip', () => {
  it('lists every member rank + sim, best first', () => {
    const span: LensSpan = {
      start: 0,
      end: 100,
      ranks: [1, 4],
      sims: [0.913, 0.7],
      bestRank: 1,
    }
    expect(formatSpanTooltip(span, 256)).toBe(
      '#1 (256t) · sim 0.91\n#4 (256t) · sim 0.70'
    )
  })
})
