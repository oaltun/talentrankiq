import { useEffect, useState } from 'react'

/**
 * Returns `value` after it has stopped changing for `delay` ms. Used to avoid firing
 * a full-table search on every keystroke.
 */
export function useDebouncedValue<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(id)
  }, [value, delay])

  return debounced
}
