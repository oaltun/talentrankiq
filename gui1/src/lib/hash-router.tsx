import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react"

const DEFAULT_PATH = "/jobs"

function parseHash(hash: string): string {
  let p = hash.startsWith("#") ? hash.slice(1) : hash
  if (p === "" || p === "/") return DEFAULT_PATH
  if (!p.startsWith("/")) p = "/" + p
  return p
}

interface HashRouterValue {
  path: string
  navigate: (path: string) => void
}

const HashRouterContext = createContext<HashRouterValue | null>(null)

export function useHashRouter(): HashRouterValue {
  const ctx = useContext(HashRouterContext)
  if (!ctx) {
    throw new Error("useHashRouter must be used within a HashRouter")
  }
  return ctx
}

export function HashRouter({ children }: { children: ReactNode }) {
  const [path, setPath] = useState(() => parseHash(window.location.hash))

  useEffect(() => {
    const onHashChange = () => {
      setPath(parseHash(window.location.hash))
    }
    window.addEventListener("hashchange", onHashChange)
    return () => window.removeEventListener("hashchange", onHashChange)
  }, [])

  const navigate = useCallback((p: string) => {
    const normalized = p.startsWith("/") ? p : "/" + p
    setPath(normalized)
    window.location.hash = "#/" + normalized.replace(/^\//, "")
  }, [])

  return (
    <HashRouterContext.Provider value={{ path, navigate }}>
      {children}
    </HashRouterContext.Provider>
  )
}
