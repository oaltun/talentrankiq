import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test for blocked job with no usable embedder.
// The job has CVs and a job ad, but no embedder is downloaded.
// The cascade should finish blocked with reason "no-active-embedder",
// keep reporting the same blocked reason with converged: false,
// and emit no repeated rank-progress done events over a several-second window.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 5

test("job with no usable embedder finishes blocked and does not loop", async ({
  page,
}) => {
  test.setTimeout(15 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-blocked-no-embedder-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-blocked-no-embedder-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-blocked-no-embedder-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-blocked-no-embedder-ad-"))

  try {
    for (let i = 0; i < CV_COUNT; i++) {
      writeFileSync(
        join(cvDir, `cv-${i + 1}.txt`),
        `Candidate ${i + 1} resume. Skills: Rust, TypeScript, React. Experience: ${i + 2} years.`
      )
    }
    const adPath = join(adDir, "jobad.txt")
    writeFileSync(
      adPath,
      "Senior Engineer. Required: Rust, TypeScript, React."
    )

    const invoke = async (
      cmd: string,
      params: Record<string, unknown>,
      timeout = 900_000
    ) => {
      const res = await page.request.post("/api/invoke", {
        data: { cmd, params },
        timeout,
      })
      expect(res.status(), `${cmd} HTTP ${res.status()}`).toBe(200)
      const body = await res.json()
      expect(body.err, `${cmd} failed: ${JSON.stringify(body)}`).toBeUndefined()
      return body.ok
    }

    // Isolated real CLI: its own app dir (jobs/db) and model cache.
    // Do NOT download any embedder — the job has no usable embedder.
    const cfg = await page.request.post("/api/test-config", {
      data: { appDir, modelDir },
    })
    expect(cfg.ok()).toBeTruthy()

    // Real GUI: create a job and open its workspace.
    await page.goto("/")
    await page.waitForLoadState("networkidle")
    await page.locator('[data-testid="add-job-button"]').click()
    await expect(
      page.locator('[data-testid^="job-item-"]').first()
    ).toBeVisible({ timeout: 15000 })
    await page
      .locator('[data-testid="job-list"]')
      .locator('[data-testid^="job-item-"]')
      .first()
      .click()
    await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
      timeout: 20000,
    })

    const jobId = await page
      .locator('[data-testid="workspace-page"]')
      .getAttribute("data-job-id")
    expect(jobId, "workspace must expose data-job-id").toBeTruthy()

    // Set job ad
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

    // Commit CV directory — this triggers processInWaves
    const input = page.locator('[data-testid="cv-dir-input"]')
    await input.click()
    await input.fill(cvDir)
    await input.press("Enter")

    // Wait for candidates to be scanned
    await expect
      .poll(
        async () => {
          const result = (await invoke("list_candidates", {
            job_id: jobId,
            offset: 0,
            limit: 100,
          })) as { total: number }
          return result.total
        },
        { timeout: 180000 }
      )
      .toBe(CV_COUNT)

    // The cascade should start and quickly finish blocked because no embedder is available.
    // Verify the blocked state via get_rank_state.
    await expect
      .poll(
        async () => {
          const st = (await invoke("get_rank_state", {
            job_id: jobId,
          })) as {
            running: boolean
            converged: boolean
            reason: string | null
          } | null
          return st
        },
        { timeout: 180000 }
      )
      .toMatchObject({
        running: false,
        converged: false,
        reason: expect.stringMatching(/^blocked:no-active-embedder$/),
      })

    // Verify job_ui_state also reports the blocked reason
    const uiState = (await invoke("job_ui_state", {
      job_id: jobId,
    })) as {
      reason: string | null
      rank?: { converged: boolean; reason: string | null }
    }
    expect(uiState.reason).toMatch(/^blocked:/)
    expect(uiState.rank?.converged).toBe(false)
    expect(uiState.rank?.reason).toMatch(/^blocked:/)

    // Verify the workspace shows the blocked status with translated label
    // and no raw "blocked:" text in the visible label
    const rankControl = page.locator('[data-testid="rank-control"]')
    await expect(rankControl).toBeVisible({ timeout: 10000 })

    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })

    // The label should show the translated reason, not the raw code
    const statusText = await statusLabel.textContent()
    expect(statusText).toBeTruthy()
    expect(statusText).not.toMatch(/^blocked:/)

    // Verify the reason is exposed via data-reason attribute
    const dataReason = await statusLabel.getAttribute("data-reason")
    expect(dataReason).toBe("no-active-embedder")

    // Verify no repeated rank-progress done events over a several-second window
    // by listening to the SSE event stream in the page
    const doneEvents: any[] = []
    await page.evaluate(() => {
      return new Promise<void>((resolve) => {
        const es = new EventSource("/api/events")
        es.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data)
            if (data.event === "rank-progress" && data.data?.stage === "done") {
              ;(window as any).__rankDoneEvents = (window as any).__rankDoneEvents || []
              ;(window as any).__rankDoneEvents.push(data.data)
            }
          } catch {}
        }
        // Wait 10 seconds to observe if any done events are emitted
        setTimeout(() => {
          es.close()
          resolve()
        }, 10000)
      })
    })

    const doneEventsCount = await page.evaluate(
      () => (window as any).__rankDoneEvents?.length ?? 0
    )
    expect(doneEventsCount).toBe(0)

    // Verify the Re-rank button is disabled with the blocked reason
    const rerankButton = rankControl.locator('[data-testid="rerank-button"]')
    await expect(rerankButton).toBeDisabled()

    const disabledReason = rankControl.locator('[data-testid="rerank-disabled-reason"]')
    const reasonText = await disabledReason.textContent()
    expect(reasonText).toBeTruthy()
    expect(reasonText).not.toMatch(/^blocked:/)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})