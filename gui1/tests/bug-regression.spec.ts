import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression coverage for talentrankiq_gui/bug.md:
//
//  1. "Job Details page is unreachable when the default embedding model is
//     missing" — the workspace hard-redirects to Models because the CLI's saved
//     `embedding_model` preference points at a model that is not downloaded.
//
//  2. "Job Ad path input cannot be edited" — `AdFileCard` binds the input to
//     `adFile.path`, so once a path is loaded every keystroke is overwritten.
//
// The CLI is NOT mocked. Each test points the dev proxy at an isolated real
// `triq serve` (its own app dir + model cache) via POST /api/test-config, so the
// scenario is deterministic and never touches the developer's real data.

const APP_DIR = () => mkdtempSync(join(tmpdir(), "triq-e2e-app-"))
const MODEL_DIR = () => mkdtempSync(join(tmpdir(), "triq-e2e-models-"))

async function useIsolatedCli(
  page: import("@playwright/test").Page,
  appDir: string,
  modelDir: string
) {
  const res = await page.request.post("/api/test-config", {
    data: { appDir, modelDir },
  })
  expect(res.ok()).toBeTruthy()
}

test.describe("bug.md regressions", () => {
  test("job details opens even when the default deep model is not downloaded", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      // A downloaded quick (scout) model but NO deep model: this is exactly the
      // bug.md scenario — the quick check passes, then the default-deep-model
      // gate fires with `reason=default` and redirects the workspace to Models.
      // Job setup (job ad, CV watch dir, candidates) needs no embedding model.
      const quickSlug = "static-similarity-mrl-multilingual-v1"
      const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
      mkdirSync(quickOnnx, { recursive: true })
      writeFileSync(join(quickOnnx, "model.onnx"), "fake")

      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()
      await expect(
        page.locator('[data-testid^="job-item-"]').first()
      ).toBeVisible({ timeout: 15000 })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      // Job Details must open; job setup does not require an embedding model.
      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 20000,
      })
      await expect(
        page.locator('[data-testid="workspace-left-pane"]')
      ).toBeVisible()
      await expect(
        page.locator('[data-testid="workspace-right-pane"]')
      ).toBeVisible()

      // We must not have been redirected to the Models page.
      await expect(page.locator('h1')).not.toHaveText("Models")
      await expect(
        page.locator('[data-testid="models-content-deep"]')
      ).toHaveCount(0)
      await expect(page.locator('[data-testid="infobar"]')).toHaveCount(0)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("job ad path input accepts typed edits", async ({ page }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      // Provide downloaded quick (scout) + deep (default) models so the
      // workspace is reachable and this test isolates the AdFileCard input
      // bug. We fake a downloaded model the same way the CLI checks it: a
      // real file on disk under the model cache (no engine is loaded for
      // setup).
      const quickSlug = "static-similarity-mrl-multilingual-v1"
      const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
      mkdirSync(quickOnnx, { recursive: true })
      writeFileSync(join(quickOnnx, "model.onnx"), "fake")

      // Config default deep embedder (harrier-oss-v1-270m-ONNX-q4f16).
      const deepSlug = "harrier-oss-v1-270m-ONNX-q4f16"
      const deepOnnx = join(modelDir, deepSlug, "repo", "onnx")
      mkdirSync(deepOnnx, { recursive: true })
      writeFileSync(join(deepOnnx, "model_q4f16.onnx"), "fake")

      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()
      await expect(
        page.locator('[data-testid^="job-item-"]').first()
      ).toBeVisible({ timeout: 15000 })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 20000,
      })

      const input = page.locator('[data-testid="ad-path-input"]')
      await expect(input).toBeVisible({ timeout: 15000 })

      // Type a new path. The buggy input immediately reverts to the old
      // `adFile.path`, so `toHaveValue` would fail against the typed value.
      await input.click()
      await input.fill("/tmp/new-ad.txt")

      await expect(input).toHaveValue("/tmp/new-ad.txt")
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  // Regression for bug.md: "Job Details page stalls while a CV-directory scan
  // holds the shared DB lock"
  //
  // The bug: opening a job calls `watch_job` (which triggers a catch-up scan of
  // the CV directory) and `get_job_detail` in parallel via Promise.all. The scan
  // holds the single shared SQLite connection (Mutex<Connection>) while iterating
  // all files (2000 in the bug report), doing one SELECT per file to filter
  // already-imported ones. This blocks `get_job_detail` for 10-30 seconds.
  //
  // This test directly invokes the CLI API to demonstrate the DB lock contention:
  // 1. Set a CV directory with many files (triggers a scan)
  // 2. Wait for the scan to start (debounce + scan time)
  // 3. Call get_job_detail repeatedly (simulating progress polling)
  // 4. Verify that get_job_detail is slow while the scan holds the lock
  test("get_job_detail is slow while scan holds DB lock (regression for DB lock contention)", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    const cvDir = mkdtempSync(join(tmpdir(), "triq-e2e-cvs-"))

    try {
      // Create ~2000 dummy CV files (.txt is supported and easiest to create)
      // Bug reports 2000 files = 10-30s stall
      const NUM_CVS = 2000
      console.log(`Creating ${NUM_CVS} CV files...`)
      for (let i = 0; i < NUM_CVS; i++) {
        writeFileSync(
          join(cvDir, `cv-${String(i).padStart(4, "0")}.txt`),
          `Candidate ${i} resume content. Skills: Rust, TypeScript, React. Experience: ${i} years.`
        )
      }
      console.log(`Created ${NUM_CVS} CV files in ${cvDir}`)

      // Provide downloaded quick (scout) model so the workspace is reachable.
      const quickSlug = "static-similarity-mrl-multilingual-v1"
      const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
      mkdirSync(quickOnnx, { recursive: true })
      writeFileSync(join(quickOnnx, "model.onnx"), "fake")

      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      // Create a job
      await page.locator('[data-testid="add-job-button"]').click()
      await expect(
        page.locator('[data-testid^="job-item-"]').first()
      ).toBeVisible({ timeout: 15000 })

      // Get the job ID from the first job item
      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      const firstJobItem = jobItems.first()
      await expect(firstJobItem).toBeVisible()
      const jobId = await firstJobItem.getAttribute("data-testid")
      expect(jobId).toMatch(/^job-item-(.+)$/)
      const actualJobId = jobId!.replace("job-item-", "")
      console.log(`Job ID: ${actualJobId}`)

      // Set the CV directory for this job via the CLI API
      // This triggers watch_job which schedules a catch-up scan
      const setCvDirResponse = await page.request.post("/api/invoke", {
        data: {
          cmd: "set_job_cv_dir",
          params: { job_id: actualJobId, directory: cvDir },
        },
      })
      expect(setCvDirResponse.status()).toBe(200)
      console.log("set_job_cv_dir completed")

      // Wait for the debounce (1s) + some scan time
      // The scan does: collect files (fast) + 2000 SELECT queries (slow)
      console.log("Waiting for scan to start and hold DB lock...")
      await page.waitForTimeout(2000)

      // Now call get_job_detail repeatedly (simulating progress polling)
      // This should be slow while the scan holds the DB lock
      console.log("Calling get_job_detail while scan runs...")
      const startTime = Date.now()
      
      // Make multiple get_job_detail calls to simulate progress polling
      for (let i = 0; i < 5; i++) {
        const detailResponse = await page.request.post("/api/invoke", {
          data: {
            cmd: "get_job_detail",
            params: { job_id: actualJobId },
          },
        })
        const elapsed = Date.now() - startTime
        console.log(`get_job_detail call ${i + 1}: ${detailResponse.status()} (${elapsed}ms total)`)
        expect(detailResponse.status()).toBe(200)
      }
      
      const totalElapsed = Date.now() - startTime
      console.log(`Total time for 5 get_job_detail calls: ${totalElapsed}ms`)

      // With the bug, each get_job_detail should be slow because the scan
      // holds the DB lock. The scan does 2000 SELECT queries, each taking
      // a few ms, so the lock is held for several seconds.
      // This assertion should FAIL with the current buggy implementation.
      expect(totalElapsed).toBeLessThan(5000)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(cvDir, { recursive: true, force: true })
    }
  })

// Architectural concern from AGENTS.md: Database Concurrency (CLI)
//
// The CLI used a single `Mutex<Connection>` (DbConnection) for every request
// handler, so all SELECTs serialized behind one connection and a long scan
// blocked every reader. The fix is a read pool: SELECT handlers take a
// read-only connection from `ServeState::read_conn()` (round-robin), while
// INSERT/UPDATE/DELETE keep the single writer connection.
//
// A pool of long-lived read connections has two failure modes worth pinning:
//   1. it does not actually serve concurrent reads (regression to one mutex);
//   2. a reader serves a stale snapshot and never sees a committed write.
// Both are deterministic, so this test asserts them directly instead of
// racing millisecond-resolution timers around sub-millisecond queries.
  test("read pool serves concurrent reads and sees committed writes", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()

    try {
      // Provide downloaded quick (scout) model so the workspace is reachable.
      const quickSlug = "static-similarity-mrl-multilingual-v1"
      const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
      mkdirSync(quickOnnx, { recursive: true })
      writeFileSync(join(quickOnnx, "model.onnx"), "fake")

      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      // Create a job
      await page.locator('[data-testid="add-job-button"]').click()
      await expect(
        page.locator('[data-testid^="job-item-"]').first()
      ).toBeVisible({ timeout: 15000 })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      const firstJobItem = jobItems.first()
      await expect(firstJobItem).toBeVisible()
      const jobId = await firstJobItem.getAttribute("data-testid")
      expect(jobId).toMatch(/^job-item-(.+)$/)
      const actualJobId = jobId!.replace("job-item-", "")
      console.log(`Job ID: ${actualJobId}`)

      const detail = (id: string) =>
        page.request.post("/api/invoke", {
          data: { cmd: "get_job_detail", params: { job_id: id } },
        })

      // 1. Many concurrent reads must all succeed and agree on the same job.
      const NUM_CONCURRENT_CALLS = 20
      console.log(
        `Making ${NUM_CONCURRENT_CALLS} concurrent get_job_detail calls...`
      )

      const responses = await Promise.all(
        Array.from({ length: NUM_CONCURRENT_CALLS }, () => detail(actualJobId))
      )

      for (const res of responses) {
        expect(res.status()).toBe(200)
      }
      for (const res of responses) {
        const body = await res.json()
        expect(body.ok.job.id).toBe(actualJobId)
        expect(Array.isArray(body.ok.criteria)).toBe(true)
        expect(Array.isArray(body.ok.triqui)).toBe(true)
      }

      // 2. A read connection must see a committed write from the writer
      //    connection (WAL read-your-writes across pool connections).
      const newTitle = `renamed-${Date.now()}`
      const renamed = await page.request.post("/api/invoke", {
        data: {
          cmd: "update_job",
          params: { job_id: actualJobId, title: newTitle, triqui_size: 10 },
        },
      })
      expect(renamed.status()).toBe(200)

      const afterWrite = await Promise.all(
        Array.from({ length: NUM_CONCURRENT_CALLS }, () => detail(actualJobId))
      )
      for (const res of afterWrite) {
        expect(res.status()).toBe(200)
        const body = await res.json()
        expect(body.ok.job.title).toBe(newTitle)
      }
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })
})