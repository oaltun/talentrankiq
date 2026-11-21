import { spawnSync } from "node:child_process"
import { existsSync } from "node:fs"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, "..")
const cliDir = resolve(repoRoot, "..", "talentrankiq_cli")
const binName = process.platform === "win32" ? "triq.exe" : "triq"
const DEV_PORT = 3000

function freePort(port: number): void {
  try {
    const result = spawnSync("lsof", ["-ti", `:${port}`], { encoding: "utf8" })
    const pids = result.stdout?.trim().split("\n").filter(Boolean) ?? []
    for (const pidStr of pids) {
      const pid = parseInt(pidStr, 10)
      if (!Number.isNaN(pid)) {
        try {
          process.kill(pid, "SIGTERM")
          console.log(`[ensure-triq] freed port ${port} by killing PID ${pid}`)
        } catch {
          // ignore
        }
      }
    }
    if (pids.length > 0) {
      new Promise((r) => setTimeout(r, 500))
    }
  } catch {
    // ignore lsof errors
  }
}

freePort(DEV_PORT)

function triqBinary(): string | null {
  if (process.env.TRIQ_BIN && existsSync(process.env.TRIQ_BIN)) {
    return process.env.TRIQ_BIN
  }
  for (const profile of ["debug", "release"]) {
    const candidate = join(cliDir, "target", profile, binName)
    if (existsSync(candidate)) return candidate
  }
  return null
}

// The Tauri shell spawns `triq serve` for every protocol command, so the CLI
// binary has to exist before the dev app starts. Building is incremental; when
// the binary is already there this is a no-op.
if (!triqBinary()) {
  console.log(`[ensure-triq] building ${cliDir} (debug)`)
  const result = spawnSync("cargo", ["build"], {
    cwd: cliDir,
    stdio: "inherit",
  })
  if (result.status !== 0) {
    console.error("[ensure-triq] cargo build failed")
    process.exit(result.status ?? 1)
  }
  if (!triqBinary()) {
    console.error(
      `[ensure-triq] triq binary still missing after build (expected target/debug/${binName})`
    )
    process.exit(1)
  }
}
