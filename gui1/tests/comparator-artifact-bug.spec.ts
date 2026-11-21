import { test, expect } from "@playwright/test"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"
import { readFileSync } from "node:fs"

const HERE = fileURLToPath(import.meta.url)
const ROOT = join(dirname(HERE), "..")

test.describe("bug.md regressions: comparator artifact & panic", () => {
  test("Cargo.toml release profile does not set panic = abort", async () => {
    const cargoTomlPath = join(ROOT, "src-tauri", "Cargo.toml")
    const cargoToml = readFileSync(cargoTomlPath, "utf8")
    expect(cargoToml).not.toContain('panic = "abort"')
  })
})
