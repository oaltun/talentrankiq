import { create } from 'zustand'

/**
 * Per-job staging for feedback. The user reorders the triqui and toggles likes here
 * (in memory); nothing hits the backend until "Apply feedback & Re-rank". `discard`
 * reverts to the snapshotted baseline; `clear` (called after Apply, once the fresh
 * `JobDetail` lands) drops the stage so it re-seeds.
 */
interface FeedbackStage {
  /** TalentRankIQ ids in stored-rank order, snapshotted at init. */
  baselineOrder: string[]
  /** User's reordered order; `null` = unchanged. */
  manualOrder: string[] | null
  /** Committed liked state at init (id -> liked). */
  baseLiked: Record<string, boolean>
  /** Staged liked overrides (id -> liked). */
  likes: Record<string, boolean>
}

interface FeedbackState {
  stages: Record<string, FeedbackStage | undefined>
  /**
   * Seed the stage from the committed triqui + liked set. A stage carrying
   * in-flight edits is never clobbered; a *clean* stage is re-seeded so a late
   * `list_liked_candidate_ids` response (e.g. the first seed after an app
   * restart) replaces the empty set the stage was initially seeded with.
   */
  init: (jobId: string, rankedIds: string[], likedIds: string[]) => void
  move: (jobId: string, id: string, dir: -1 | 1) => void
  toggleLike: (jobId: string, id: string) => void
  discard: (jobId: string) => void
  clear: (jobId: string) => void
}

/** Whether a stage carries unstaged user edits (manual order or like toggles). */
function isStageDirty(stage: FeedbackStage): boolean {
  if (stage.manualOrder) return true
  const keys = new Set([
    ...Object.keys(stage.baseLiked),
    ...Object.keys(stage.likes),
  ])
  for (const id of keys) {
    if (!!stage.likes[id] !== !!stage.baseLiked[id]) return true
  }
  return false
}

export const useFeedbackStore = create<FeedbackState>(set => ({
  stages: {},
  init: (jobId, rankedIds, likedIds) =>
    set(state => {
      // Keep in-flight edits; only a clean stage may be re-seeded (the stage
      // is created as soon as the triqui lands — possibly before the
      // committed liked set has loaded — so re-seeding when the liked set
      // arrives is what makes stars survive an app restart).
      const existing = state.stages[jobId]
      if (existing && isStageDirty(existing)) return state
      const baseLiked: Record<string, boolean> = {}
      for (const id of likedIds) baseLiked[id] = true
      return {
        stages: {
          ...state.stages,
          [jobId]: {
            baselineOrder: rankedIds,
            manualOrder: null,
            baseLiked,
            likes: { ...baseLiked },
          },
        },
      }
    }),
  move: (jobId, id, dir) =>
    set(state => {
      const stage = state.stages[jobId]
      if (!stage) return state
      const order = stage.manualOrder ?? stage.baselineOrder
      const idx = order.indexOf(id)
      if (idx < 0) return state
      const target = idx + dir
      if (target < 0 || target >= order.length) return state
      const x = order[idx]
      const y = order[target]
      if (x === undefined || y === undefined) return state
      const next = [...order]
      next[idx] = y
      next[target] = x
      return {
        stages: { ...state.stages, [jobId]: { ...stage, manualOrder: next } },
      }
    }),
  toggleLike: (jobId, id) =>
    set(state => {
      const stage = state.stages[jobId]
      if (!stage) return state
      const cur = stage.likes[id] ?? false
      return {
        stages: {
          ...state.stages,
          [jobId]: { ...stage, likes: { ...stage.likes, [id]: !cur } },
        },
      }
    }),
  discard: jobId =>
    set(state => {
      const stage = state.stages[jobId]
      if (!stage) return state
      return {
        stages: {
          ...state.stages,
          [jobId]: {
            ...stage,
            manualOrder: null,
            likes: { ...stage.baseLiked },
          },
        },
      }
    }),
  clear: jobId =>
    set(state => ({
      stages: Object.fromEntries(
        Object.entries(state.stages).filter(([key]) => key !== jobId)
      ),
    })),
}))

// --- selectors (use via `useFeedbackStore(selectX(jobId))`; selector syntax only) ---

/** Stable empty map so `selectLikes` never returns a fresh reference (Zustand v5
 * re-render loop otherwise). */
const EMPTY_LIKES: Record<string, boolean> = {}

/** The staged triqui order, or `null` when unchanged. */
export function selectOrder(jobId: string) {
  return (s: FeedbackState): string[] | null =>
    s.stages[jobId]?.manualOrder ?? null
}

/** The effective liked map (committed + staged toggles). */
export function selectLikes(jobId: string) {
  return (s: FeedbackState): Record<string, boolean> =>
    s.stages[jobId]?.likes ?? EMPTY_LIKES
}

/** Whether the stage has unstaged edits. */
export function selectIsDirty(jobId: string) {
  return (s: FeedbackState): boolean => {
    const stage = s.stages[jobId]
    return stage ? isStageDirty(stage) : false
  }
}
