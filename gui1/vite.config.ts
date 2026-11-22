import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { resolve } from "node:path"
import process from "node:process"
import { existsSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

function resolveTriqBin(): string {
  if (process.env.TRIQ_BIN) return process.env.TRIQ_BIN
  const here = dirname(fileURLToPath(import.meta.url))
  const candidates = [
    join(here, "..", "cli1", "target", "debug", "triq"),
    join(here, "..", "cli1", "target", "release", "triq"),
  ]
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate
  }
  return "triq"
}

// Tauri's app_data_dir for the identifier in src-tauri/tauri.conf.json. The
// dev proxy must resolve the same directory, otherwise browser e2e runs and the
// desktop app keep separate DBs and preferences.
const APP_IDENTIFIER = "com.razalt.talentrankiq"

function tauriAppDataDir(): string {
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", APP_IDENTIFIER)
  }
  if (process.platform === "win32") {
    return join(
      process.env.APPDATA ?? join(homedir(), "AppData", "Roaming"),
      APP_IDENTIFIER
    )
  }
  return join(
    process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share"),
    APP_IDENTIFIER
  )
}

function triqProxyPlugin() {
  let child: any = null
  let nextId = 1
  // Overridable per-test so e2e specs can point the proxy at an isolated
  // real `triq serve` (own app dir + model cache) via POST /api/test-config.
  let appDir = process.env.TRIQ_APP_DIR
  let modelDir = process.env.TRIQI_MODEL_DIR
  const pending = new Map<
    number,
    { resolve: (v: any) => void; reject: (e: any) => void }
  >()
  const sseClients = new Set<any>()
  let stdoutBuf = ""

  function killChild() {
    if (child && !child.killed) {
      try {
        child.kill("SIGKILL")
      } catch {
        // ignore
      }
    }
    // Reject anything still waiting on the old child, otherwise its `close`
    // handler (which may fire after a replacement was already spawned) would
    // reject a *newer* child's pending requests with "triq serve exited".
    for (const [id, p] of pending) {
      p.reject(new Error("triq serve restarted"))
      pending.delete(id)
    }
    child = null
  }

  function broadcastEvent(line: string) {
    for (const res of sseClients) {
      try {
        res.write(`data: ${line}\n\n`)
      } catch {
        sseClients.delete(res)
      }
    }
  }

  async function ensureChild() {
    if (child && !child.killed) return child
    const triq = resolveTriqBin()
    const { spawn } = await import("node:child_process")
    const env: Record<string, string> = { ...process.env } as any
    // TRIQ_APP_DIR / TRIQI_MODEL_DIR can be overridden by the e2e control route
    // below so a test drives an isolated real `triq serve` instead of the dev's
    // home-directory state.
    env.TRIQ_APP_DIR = appDir ?? join(tauriAppDataDir(), "triq")
    if (modelDir) {
      env.TRIQI_MODEL_DIR = modelDir
    } else {
      delete env.TRIQI_MODEL_DIR
    }
    const proc = spawn(triq, ["serve"], { env, stdio: ["pipe", "pipe", "inherit"] })
    child = proc
    proc.on("error", (e: any) => {
      // A superseded child (killed by /api/test-config) can still emit late
      // events; those must not disturb the replacement's pending requests.
      if (child !== proc) return
      const msg = e?.message ?? String(e)
      console.error(`[triq-proxy] failed to spawn ${triq}: ${msg}`)
      for (const [id, p] of pending) {
        p.reject(new Error(`triq spawn failed: ${msg}`))
        pending.delete(id)
      }
      child = null
    })
    stdoutBuf = ""
    proc.stdout.on("data", (d: any) => {
      stdoutBuf += d.toString()
      let idx
      while ((idx = stdoutBuf.indexOf("\n")) >= 0) {
        const line = stdoutBuf.slice(0, idx).trim()
        stdoutBuf = stdoutBuf.slice(idx + 1)
        if (!line) continue
        try {
          const j = JSON.parse(line)
          if (typeof j.id === "number" && pending.has(j.id)) {
            const p = pending.get(j.id)!
            pending.delete(j.id)
            p.resolve(j)
          } else if (typeof j.event === "string") {
            broadcastEvent(line)
          }
        } catch {
          // ignore non-json
        }
      }
    })
    proc.on("close", () => {
      if (child !== proc) return
      for (const [id, p] of pending) {
        p.reject(new Error("triq serve exited"))
        pending.delete(id)
      }
      child = null
    })
    return proc
  }

  return {
    name: "triq-proxy",
    configureServer(server: any) {
      // E2E control route: point the proxy at a fresh, isolated real `triq serve`
      // (its own app dir + model cache) and restart the child. This lets a test
      // reproduce a CLI state (e.g. a stale default model) deterministically
      // without touching the developer's real home-directory data. The app and
      // CLI are NOT mocked: the proxy still spawns and speaks to the real binary.
      server.middlewares.use("/api/test-config", (req: any, res: any) => {
        if (req.method !== "POST") {
          res.statusCode = 405
          res.end()
          return
        }
        let body = ""
        req.on("data", (c: any) => {
          body += c
        })
        req.on("end", () => {
          try {
            const parsed = JSON.parse(body || "{}")
            if (typeof parsed.appDir === "string" && parsed.appDir) {
              appDir = parsed.appDir
            }
            if (typeof parsed.modelDir === "string" && parsed.modelDir) {
              modelDir = parsed.modelDir
            }
            killChild()
            res.setHeader("Content-Type", "application/json")
            res.end(JSON.stringify({ ok: true, appDir, modelDir }))
          } catch (e: any) {
            res.statusCode = 500
            res.end(JSON.stringify({ err: { message: String(e) } }))
          }
        })
      })
      server.middlewares.use("/api/events", (req: any, res: any) => {
        res.statusCode = 200
        res.setHeader("Content-Type", "text/event-stream")
        res.setHeader("Cache-Control", "no-cache, no-transform")
        res.setHeader("Connection", "keep-alive")
        res.setHeader("Access-Control-Allow-Origin", "*")
        res.flushHeaders?.()
        res.write("retry: 1000\n\n")
        sseClients.add(res)
        req.on("close", () => {
          sseClients.delete(res)
        })
      })
      server.middlewares.use("/api/invoke", (req: any, res: any) => {
        if (req.method !== "POST") {
          res.statusCode = 405
          res.end()
          return
        }
        let body = ""
        req.on("data", (c: any) => {
          body += c
        })
        req.on("end", async () => {
          const startedAt = Date.now()
          try {
            const parsed = JSON.parse(body || "{}")
            const cmd = parsed.cmd
            const params = parsed.params ?? {}
            console.log(`[triq-proxy] -> ${cmd}`)
            const c = await ensureChild()
            const id = nextId++
            const resp = await new Promise<any>((resolve, reject) => {
              pending.set(id, { resolve, reject })
              try {
                c.stdin.write(JSON.stringify({ id, cmd, params }) + "\n")
              } catch (e) {
                pending.delete(id)
                reject(e)
              }
            })
            res.setHeader("Content-Type", "application/json")
            res.end(JSON.stringify(resp))
            console.log(`[triq-proxy] <- ${cmd} ${Date.now() - startedAt}ms`)
          } catch (e: any) {
            res.statusCode = 500
            res.setHeader("Content-Type", "application/json")
            res.end(JSON.stringify({ err: { message: String(e) } }))
          }
        })
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), triqProxyPlugin()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/target/**"],
    },
  },
})
