import { describe, expect, it, beforeEach } from 'vitest'

import {
  selectIsDirty,
  selectLikes,
  selectOrder,
  useFeedbackStore,
} from './feedback-store'

const JOB = 'job_1'

function seed(order = ['a', 'b', 'c'], liked: string[] = []) {
  useFeedbackStore.getState().clear(JOB)
  useFeedbackStore.getState().init(JOB, order, liked)
}

describe('feedback-store', () => {
  beforeEach(() => seed())

  it('is clean after seeding; order falls back to baseline', () => {
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toBeNull()
    expect(selectLikes(JOB)(useFeedbackStore.getState())).toEqual({})
  })

  it('move swaps neighbours and marks dirty', () => {
    useFeedbackStore.getState().move(JOB, 'b', -1)
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toEqual([
      'b',
      'a',
      'c',
    ])
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(true)
  })

  it('move clamps at the top boundary (no-op, stays dirty only if moved)', () => {
    useFeedbackStore.getState().move(JOB, 'a', -1) // already first
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toBeNull()
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
  })

  it('move clamps at the bottom boundary', () => {
    useFeedbackStore.getState().move(JOB, 'c', 1) // already last
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toBeNull()
  })

  it('toggleLike flips and marks dirty', () => {
    useFeedbackStore.getState().toggleLike(JOB, 'a')
    expect(selectLikes(JOB)(useFeedbackStore.getState()).a).toBe(true)
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(true)
  })

  it('toggleLike twice cancels (clean again)', () => {
    useFeedbackStore.getState().toggleLike(JOB, 'a')
    useFeedbackStore.getState().toggleLike(JOB, 'a')
    expect(selectLikes(JOB)(useFeedbackStore.getState()).a).toBeFalsy()
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
  })

  it('discard reverts order + likes to the seeded baseline', () => {
    useFeedbackStore.getState().move(JOB, 'c', -1)
    useFeedbackStore.getState().toggleLike(JOB, 'b')
    useFeedbackStore.getState().discard(JOB)
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toBeNull()
    expect(selectLikes(JOB)(useFeedbackStore.getState()).b).toBeFalsy()
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
  })

  it('seeds liked ids and reports them as clean', () => {
    seed(['a', 'b'], ['b'])
    expect(selectLikes(JOB)(useFeedbackStore.getState()).b).toBe(true)
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
  })

  it('init keeps in-flight edits (does not clobber a dirty stage)', () => {
    useFeedbackStore.getState().move(JOB, 'b', -1)
    // a fresh detail arrives (same job) — edits must survive
    useFeedbackStore.getState().init(JOB, ['a', 'b', 'c'], [])
    expect(selectOrder(JOB)(useFeedbackStore.getState())).toEqual([
      'b',
      'a',
      'c',
    ])
  })

  it('init re-seeds a clean stage when the committed liked set arrives late', () => {
    // App restart: the stage is seeded from the triqui before
    // list_liked_candidate_ids resolves, so the committed set is empty…
    seed(['a', 'b'], [])
    // …and when the real liked set lands, the clean stage must pick it up
    // (this is what makes committed stars visible after a restart).
    useFeedbackStore.getState().init(JOB, ['a', 'b'], ['b'])
    expect(selectLikes(JOB)(useFeedbackStore.getState()).b).toBe(true)
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(false)
  })

  it('init keeps staged like edits when a fresh liked set arrives', () => {
    seed(['a', 'b'], [])
    useFeedbackStore.getState().toggleLike(JOB, 'a') // dirty: not yet applied
    useFeedbackStore.getState().init(JOB, ['a', 'b'], ['b'])
    const likes = selectLikes(JOB)(useFeedbackStore.getState())
    expect(likes.a).toBe(true) // staged edit survives
    expect(likes.b).toBeFalsy() // not re-seeded over the dirty stage
    expect(selectIsDirty(JOB)(useFeedbackStore.getState())).toBe(true)
  })

  it('init re-seeds a stage that was toggled back to clean', () => {
    seed(['a', 'b'], [])
    useFeedbackStore.getState().toggleLike(JOB, 'a')
    useFeedbackStore.getState().toggleLike(JOB, 'a') // cancels out → clean
    useFeedbackStore.getState().init(JOB, ['a', 'b'], ['b'])
    expect(selectLikes(JOB)(useFeedbackStore.getState()).b).toBe(true)
  })
})
