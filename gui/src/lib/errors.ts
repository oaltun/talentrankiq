/**
 * Rendering for whatever a command can throw.
 *
 * `unwrapResult` (see `tauri-bindings`) throws `result.error`, which for domain
 * errors is a tagged plain object — `{ type: 'ValidationError', message }`, or
 * `{ type: 'NotFound', entity, id }` — not an `Error`. So both `String(e)` and
 * `e instanceof Error ? e.message : String(e)` render "[object Object]" and
 * throw away the already-good backend message. Read the tag instead.
 *
 * Structural types, not the generated `DomainError`: this module must stay
 * importable without the codegen output so it can be unit-tested on its own.
 */

interface TaggedDomainError {
  type?: string
  entity?: string
  id?: string
  message?: string
}

/** Human-readable text for an unknown thrown value. */
export function errorMessage(e: unknown): string {
  if (typeof e === 'string') return e
  if (e instanceof Error) return e.message
  if (e && typeof e === 'object') {
    const d = e as TaggedDomainError
    if (d.type === 'NotFound') {
      return `${d.entity ?? 'Item'} not found: ${d.id ?? ''}`.trim()
    }
    if (typeof d.message === 'string' && d.message) return d.message
  }
  return String(e)
}

/**
 * True when `e` is the typed "this entity does not exist" domain error for
 * `entity`. Branching on the tag (never on the message text) keeps call sites
 * correct across a message rewording — e.g. the gui repairs a missing
 * `CV directory` in one click instead of showing a dead-end failure.
 */
export function isDomainNotFound(e: unknown, entity: string): boolean {
  if (!e || typeof e !== 'object') return false
  const d = e as TaggedDomainError
  return d.type === 'NotFound' && d.entity === entity
}
