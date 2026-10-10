import { describe, expect, it } from 'vitest'
import { fmtScore, fmtWeight, per100, scoringFormula } from './scores'

describe('per100', () => {
  it('rescales a question grade to 0..100', () => {
    // FB1 stored 8.28 points against a weight of 10 → 82.8.
    expect(per100(8.28, 10)).toBeCloseTo(82.8, 6)
  })

  it('makes the total the weighted average of the grades, out of 100', () => {
    // job-10: judge 44.79 + FB1 8.28 + FB2 4.14 = 57.21 of a 65 rubric.
    // The Average is the weighted *mean* of the grades, which is the same as
    // dividing the summed weighted points by the summed weights — so weights
    // that do not add up to 100 still yield a Average out of 100.
    const terms = [
      [44.79, 50],
      [8.28, 10],
      [4.14, 5],
    ] as const
    const weightSum = terms.reduce((a, [, w]) => a + w, 0)
    const pointsSum = terms.reduce((a, [p]) => a + p, 0)
    const gradeOf = (points: number, weight: number) => (points / weight) * 100
    const weightedAverage =
      terms.reduce((a, [p, w]) => a + gradeOf(p, w) * w, 0) / weightSum

    expect(weightSum).toBe(65)
    expect(per100(pointsSum, weightSum)).toBeCloseTo(weightedAverage, 6)
    expect(per100(pointsSum, weightSum)).toBeCloseTo(88.015384615, 6)
  })

  it('keeps a negative criterion a plain 0..100 grade (its sign is the weight)', () => {
    expect(per100(-7, -10)).toBeCloseTo(70, 6)
  })

  it('reads the judge row as its literal win-rate', () => {
    // The raw is the tournament win-rate untransformed, so the grade IS the
    // win-rate percentage (thin-gui.md, judge rubric row).
    expect(per100(50, 50)).toBeCloseTo(100, 6)
    expect(per100(44.79, 50)).toBeCloseTo(89.58, 6)
    expect(per100(30, 50)).toBeCloseTo(60, 6)
    // A CV that wins nothing grades 0, never negative.
    expect(per100(0, 50)).toBeCloseTo(0, 6)
  })

  it('has no grade without points or without a scale to read them against', () => {
    expect(per100(null, 10)).toBeNull()
    expect(per100(undefined, 10)).toBeNull()
    expect(per100(5, null)).toBeNull()
    expect(per100(0, 0)).toBeNull()
    expect(per100(5, 1e-12)).toBeNull()
    expect(per100(Number.NaN, 10)).toBeNull()
  })
})

describe('fmtScore', () => {
  it('keeps one decimal and renders a dash for nothing', () => {
    expect(fmtScore(4)).toBe('4.0')
    expect(fmtScore(82.83)).toBe('82.8')
    expect(fmtScore(-92.66)).toBe('-92.7')
    expect(fmtScore(null)).toBe('–')
    expect(fmtScore(undefined)).toBe('–')
  })
})

describe('fmtWeight', () => {
  it('drops a pointless decimal but keeps a real one', () => {
    expect(fmtWeight(50)).toBe('50')
    expect(fmtWeight(2.5)).toBe('2.5')
    expect(fmtWeight(-10)).toBe('-10')
  })
})

describe('scoringFormula', () => {
  it('spells the Average out with the weights of the given job', () => {
    // job-10: ad 50, FB1 10, FB2 5, no criteria.
    expect(
      scoringFormula({ ad_weight: 50, fb_job_weight: 10, fb_profile_weight: 5 })
    ).toBe('Average = 50 × ad + 10 × FB1 + 5 × FB2')
  })

  it('names a shared criterion weight once', () => {
    expect(
      scoringFormula(
        { ad_weight: 50, fb_job_weight: 10, fb_profile_weight: 5 },
        [{ weight: 10 }, { weight: 10 }]
      )
    ).toBe('Average = 50 × ad + 10 × criteria + 10 × FB1 + 5 × FB2')
  })

  it('shows a range when criteria weights differ', () => {
    expect(
      scoringFormula({ ad_weight: 50 }, [{ weight: 10 }, { weight: 25 }])
    ).toBe('Average = 50 × ad + criteria 10–25 each')
  })

  it('drops zero-weight questions and copes with an empty rubric', () => {
    expect(scoringFormula({ ad_weight: 50, fb_job_weight: 0 })).toBe(
      'Average = 50 × ad'
    )
    expect(scoringFormula({})).toBe('Average = —')
  })
})
