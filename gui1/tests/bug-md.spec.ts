import { test, expect, type Page } from "@playwright/test"
import {
  mkdtempSync,
  writeFileSync,
  mkdirSync,
  rmSync,
  readFileSync,
} from "node:fs"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

// Regression tests for talentrankiq_gui/bug.md (the current bug report):
//
//  2. CandidatesPane renders rows without a React `key`.
//  3. The e2e suite is not isolated from the developer's CLI app dir.
//  4. Candidates imported in the background never appear when the pane already
//     has candidates (polling only runs while `total === 0`).
//
// The CLI is NOT mocked: each test points the dev proxy at an isolated real
// `triq serve` via POST /api/test-config, then drives the real UI.

const HERE = dirname(fileURLToPath(import.meta.url))
const APP_DIR = () => mkdtempSync(join(tmpdir(), "triq-bugmd-app-"))
const MODEL_DIR = () => mkdtempSync(join(tmpdir(), "triq-bugmd-models-"))
const CV_DIR = () => mkdtempSync(join(tmpdir(), "triq-bugmd-cvs-"))

async function useIsolatedCli(page: Page, appDir: string, modelDir: string) {
  const res = await page.request.post("/api/test-config", {
    data: { appDir, modelDir },
  })
  expect(res.ok()).toBeTruthy()
}

// A downloaded quick (scout) model is enough for job setup: the workspace no
// longer redirects when the default deep model is missing.
function seedQuickModel(modelDir: string) {
  const quickSlug = "static-similarity-mrl-multilingual-v1"
  const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
  mkdirSync(quickOnnx, { recursive: true })
  writeFileSync(join(quickOnnx, "model.onnx"), "fake")
}

async function openFirstJobWorkspace(page: Page) {
  await page.goto("/")
  await page.waitForLoadState("networkidle")

  await page.locator('[data-testid="add-job-button"]').click()
  await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
    timeout: 15000,
  })

  await page
    .locator('[data-testid="job-list"]')
    .locator('[data-testid^="job-item-"]')
    .first()
    .click()

  await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
    timeout: 20000,
  })
}

test.describe("bug.md regressions", () => {
  // Issue 2: "Missing React Keys in CandidatesPane".
  //
  // `renderCandidateRow` returns a bare `<tr>`, so React logs
  // "Each child in a list should have a unique key prop" and reuses DOM nodes
  // when the list order changes. Capture the console while real imported
  // candidates render.
  test("candidate rows do not trigger a React unique-key warning", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    const cvDir = CV_DIR()

    try {
      for (let i = 0; i < 5; i++) {
        writeFileSync(
          join(cvDir, `cv-${i}.txt`),
          `Candidate ${i} resume. Skills: Rust, TypeScript, React. Experience: ${i} years.`
        )
      }
      seedQuickModel(modelDir)

      const keyWarnings: string[] = []
      page.on("console", (msg) => {
        if (
          msg.type() === "error" &&
          /unique\s+"?key"?\s+prop/i.test(msg.text())
        ) {
          keyWarnings.push(msg.text())
        }
      })

      await useIsolatedCli(page, appDir, modelDir)
      await openFirstJobWorkspace(page)

      const cvDirInput = page.locator('[data-testid="cv-dir-input"]')
      await cvDirInput.click()
      await cvDirInput.fill(cvDir)
      await cvDirInput.press("Enter")

      const candidateRows = page.locator(
        '[data-testid="candidates-pane"] tbody tr'
      )
      await expect(candidateRows.first()).toBeVisible({ timeout: 60000 })
      expect(await candidateRows.count()).toBeGreaterThan(1)

      expect(
        keyWarnings,
        `React key warnings:\n${keyWarnings.join("\n")}`
      ).toEqual([])
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(cvDir, { recursive: true, force: true })
    }
  })

  // Issue 3: "E2E Test Isolation Failure".
  //
  // `flow.spec.ts` drives the real CLI app dir, and `test_fast.sh` only cleans
  // `com.talentrankiq.gui` while the app/CLI identifier is
  // `com.razalt.talentrankiq` (vite.config.ts / tauri.conf.json). A second
  // `npm test` therefore inherits jobs and CV state from the first run.
  test("e2e suite is isolated from the real CLI app dir", async () => {
    const viteConfig = readFileSync(join(HERE, "../vite.config.ts"), "utf8")
    const match = viteConfig.match(/const APP_IDENTIFIER = "([^"]+)"/)
    expect(match, "vite.config.ts must declare APP_IDENTIFIER").not.toBeNull()
    const appIdentifier = match![1]

    const testFast = readFileSync(join(HERE, "../test_fast.sh"), "utf8")
    // test_fast.sh must remove the real CLI app-data/cache dirs, not just
    // mention the identifier in an unrelated DB-cleanup line. Currently it
    // only cleans `com.talentrankiq.gui`, so jobs survive between runs.
    expect(
      testFast,
      `test_fast.sh must clean Application Support/${appIdentifier}`
    ).toContain(`Application Support/${appIdentifier}`)
    expect(
      testFast,
      `test_fast.sh must clean Caches/${appIdentifier}`
    ).toContain(`Caches/${appIdentifier}`)

    const flowSpec = readFileSync(join(HERE, "flow.spec.ts"), "utf8")
    expect(
      flowSpec,
      "flow.spec.ts must call useIsolatedCli() so it never touches real data"
    ).toMatch(/useIsolatedCli\s*\(/)
  })

  // Issue 4: "Candidates Not Appearing After Import".
  //
  // bug.md claims the fix is polling plus rank-progress events. In the code,
  // CandidatesPane only polls while `total === 0`, and the rank-progress
  // listener lives in SetupPane (progress bars only). So when a job already
  // shows candidates and a *new* CV directory import lands in the background,
  // the table never reloads: exactly the described case "initial load returned
  // total > 0 but candidates were still arriving".
  test("candidates refresh after a background import when the pane already has candidates", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    const firstCvDir = CV_DIR()
    const secondCvDir = CV_DIR()

    const writeCvs = (dir: string, prefix: string, count: number) => {
      for (let i = 0; i < count; i++) {
        writeFileSync(
          join(dir, `${prefix}-${i}.txt`),
          `${prefix} candidate ${i} resume. Skills: Rust, TypeScript, React. Experience: ${i + 1} years.`
        )
      }
    }

    try {
      writeCvs(firstCvDir, "alpha", 3)
      writeCvs(secondCvDir, "beta", 6)
      seedQuickModel(modelDir)

      await useIsolatedCli(page, appDir, modelDir)
      await openFirstJobWorkspace(page)

      const jobId = await page
        .locator('[data-testid="workspace-page"]')
        .getAttribute("data-job-id")
      expect(jobId, "workspace must expose data-job-id").toBeTruthy()

      const listCliCandidates = async () => {
        const res = await page.request.post("/api/invoke", {
          data: {
            cmd: "list_candidates",
            params: { job_id: jobId, offset: 0, limit: 100 },
          },
        })
        expect(res.status()).toBe(200)
        const body = await res.json()
        expect(body.err, JSON.stringify(body)).toBeUndefined()
        return body.ok as {
          items: Array<{ file_name: string }>
          total: number
        }
      }

      const commitCvDir = async (dir: string) => {
        const input = page.locator('[data-testid="cv-dir-input"]')
        await input.click()
        await input.fill(dir)
        await input.press("Enter")
      }

      // First import: the pane starts empty, so the `total === 0` polling
      // shows these candidates.
      await commitCvDir(firstCvDir)
      await expect
        .poll(async () => (await listCliCandidates()).total, { timeout: 60000 })
        .toBe(3)
      await expect(
        page.locator('[data-testid="candidates-pane"] tbody tr')
      ).toHaveCount(3, { timeout: 60000 })

      // Second import: the pane already has total > 0, so its polling is off.
      // The import still runs in the background and lands new candidates.
      await commitCvDir(secondCvDir)
      await expect
        .poll(async () => (await listCliCandidates()).total, { timeout: 60000 })
        .toBe(9)

      // The beta-* candidates are in the CLI/DB now. The pane must pick them
      // up without a reload; currently it keeps showing only the alpha rows.
      await expect(
        page.getByText("beta-0.txt", { exact: true }),
        "candidates pane should show candidates imported in the background"
      ).toBeVisible({ timeout: 20000 })
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(firstCvDir, { recursive: true, force: true })
      rmSync(secondCvDir, { recursive: true, force: true })
    }
  })
})
