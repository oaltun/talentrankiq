/**
 * Score scales for the results sheet.
 *
 * The backend stores, per rubric question, `points = weight × raw` alongside
 * `max_points` = that question's weight, and a candidate's `points` is the sum
 * over the questions that count (with `points_max` their summed weights). The
 * weights are the rubric's *multipliers* into the Average, not per-column
 * ceilings, so the sheet shows everything on one common scale instead:
 *
 *     grade = points / max_points × 100        (one question, out of 100)
 *     mean  = points / points_max × 100        (the row, out of 100)
 *
 * The Average is therefore the weighted *mean* of the grades. That is what
 * lets the weights stay un-normalised inputs (50 / 10 / 5 need not sum to 100)
 * while the Average is always out of 100 — and it is a positive rescale of the
 * stored points, so rank order is untouched and no re-rank is needed to adopt it.
 *
 * Dividing by the *signed* max keeps a negative criterion's grade a plain
 * 0..100 similarity (its sign lives in the weight) — so every cell in the sheet
 * is a 0..100 grade, including the judge row, whose raw is the tournament
 * win-rate: a displayed 70 there means "wins 70% of its comparisons".
 */

/** One question's (or a row's) stored points rescaled to the shared 0..100
 * grade scale. `null` = nothing to show: no score at all, or a zero weight, so
 * there is no scale to read the points against. */
export function per100(
  points: number | null | undefined,
  max: number | null | undefined
): number | null {
  if (points == null || max == null || !Number.isFinite(points)) return null
  if (!Number.isFinite(max) || Math.abs(max) < 1e-9) return null
  return (points / max) * 100
}

/** A grade for display: one decimal always kept (4 renders as "4.0"), `–` for
 * "not graded yet". */
export function fmtScore(n: number | null | undefined): string {
  return n == null ? '–' : (Math.round(n * 10) / 10).toFixed(1)
}

/** A rubric weight for display, dropping a pointless `.0` (50, not 50.0). */
export function fmtWeight(w: number): string {
  return Number.isInteger(w) ? String(w) : String(Math.round(w * 10) / 10)
}

/** The structural slice of `Job` a formula needs — kept structural so the real
 * DTO satisfies it without an import cycle. */
export interface RubricWeights {
  ad_weight?: number | null
  fb_job_weight?: number | null
  fb_profile_weight?: number | null
}

/**
 * The Average spelled out with this job's own weights, e.g.
 * `Average = 50 × ad + 10 × FB1 + 5 × FB2` (each score read out of 100).
 *
 * The weights are the configured multipliers even though they need not add up
 * to 100 — the Average is their weighted *mean*, so it is always out of 100.
 * Zero-weight questions are dropped (they contribute nothing); differing
 * criterion weights are shown as a range rather than pretending to be one
 * multiplier.
 */
export function scoringFormula(
  job: RubricWeights,
  criteria: readonly { weight: number }[] = []
): string {
  const terms: string[] = []
  const add = (label: string, weight: number | null | undefined) => {
    if (weight == null || !Number.isFinite(weight)) return
    if (Math.abs(weight) < 1e-9) return
    terms.push(`${fmtWeight(weight)} × ${label}`)
  }
  add('ad', job.ad_weight)
  const distinct = [...new Set(criteria.map(c => c.weight))]
  if (distinct.length === 1) add('criteria', distinct[0])
  else if (distinct.length > 1) {
    terms.push(
      `criteria ${fmtWeight(Math.min(...distinct))}–${fmtWeight(
        Math.max(...distinct)
      )} each`
    )
  }
  add('FB1', job.fb_job_weight)
  add('FB2', job.fb_profile_weight)
  return terms.length > 0 ? `Average = ${terms.join(' + ')}` : 'Average = —'
}
