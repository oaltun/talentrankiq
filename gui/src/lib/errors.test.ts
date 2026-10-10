import { describe, it, expect } from 'vitest'

import { errorMessage, isDomainNotFound } from './errors'

describe('errorMessage', () => {
  // The bug this exists for: a domain error is a tagged plain object, so the
  // obvious `String(e)` renders the user with "[object Object]".
  it('renders a ValidationError message instead of [object Object]', () => {
    expect(
      errorMessage({
        type: 'ValidationError',
        message: 'Not a directory: /Users/me/jobs/job-1',
      })
    ).toBe('Not a directory: /Users/me/jobs/job-1')
  })

  it('renders a NotFound as "entity not found: id"', () => {
    expect(
      errorMessage({
        type: 'NotFound',
        entity: 'CV directory',
        id: '/Users/me/jobs/job-1',
      })
    ).toBe('CV directory not found: /Users/me/jobs/job-1')
  })

  it('passes a plain Error message and a string through', () => {
    expect(errorMessage(new Error('boom'))).toBe('boom')
    expect(errorMessage('already text')).toBe('already text')
  })

  it('falls back rather than printing [object Object] for unknown shapes', () => {
    expect(errorMessage(undefined)).toBe('undefined')
    expect(errorMessage(42)).toBe('42')
    // Tagged but message-less: String() would still say [object Object].
    expect(errorMessage({ type: 'IoError' })).toBe('[object Object]')
  })
})

describe('isDomainNotFound', () => {
  it('branches on the tag, not the message text', () => {
    expect(
      isDomainNotFound(
        { type: 'NotFound', entity: 'CV directory', id: '/x' },
        'CV directory'
      )
    ).toBe(true)
    expect(
      isDomainNotFound(
        { type: 'NotFound', entity: 'Job', id: 'j1' },
        'CV directory'
      )
    ).toBe(false)
    expect(
      isDomainNotFound(
        { type: 'ValidationError', message: 'CV directory gone' },
        'CV directory'
      )
    ).toBe(false)
    expect(isDomainNotFound(new Error('x'), 'CV directory')).toBe(false)
    expect(isDomainNotFound(null, 'CV directory')).toBe(false)
  })
})
