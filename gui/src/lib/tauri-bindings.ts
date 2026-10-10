// Re-export the generated tauri-specta bindings and provide the `unwrapResult`
// helper. The generated content lives in `bindings.ts` (git-ignored for
// prettier/eslint); keeping `unwrapResult` here means regenerating the bindings
// never wipes it.

export * from './bindings'

// Helper to unwrap a tauri-specta `Result`.
export function unwrapResult<T, E>(result: import('./bindings').Result<T, E>): T {
  if (result.status === 'ok') return result.data
  throw result.error
}

// Note: `unwrapResult` throws a tagged domain-error *object*, not an Error.
// Render/branch on it with `errorMessage` / `isDomainNotFound` from `./errors`
// — deliberately NOT re-exported here, so importing the helpers never drags
// the generated bindings into a module graph (which breaks unit tests).
