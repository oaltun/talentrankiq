import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression test for fix-blocked-job-recovery.
//
// A job whose cascade finishes blocked (here: missing-rubric-vectors, because
// no embedder is available yet) used to be treated as "running" by the
// backend: `spawn_cascade` refused to start a new run, and `job_ui_state` /
// `get_rank_state` reported `running = true`, so the Re-rank button stayed
// disabled. Once the blocking condition was resolved (an embedder becomes
// available) the job could never recover without manual DB surgery.
//
// No mocks: real GUI, real `triq serve`, real upstream model downloads, and
// the real Re-rank button. The flow:
//   1. Reach a blocked state (no embedder downloaded).
//   2. Assert the backend reports running = false and Re-rank is enabled.
//   3. Download the embedders (resolve the blocking condition).
//   4. Click Re-rank and assert a new cascade starts and the job converges.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 5

test("blocked job recovers via Re-rank after its inputs allow ranking", async ({
  page,
}) => {
  // Model downloads + first-time ORT session load dominate the runtime.
  test.setTimeout(30 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-blocked-recovery-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-blocked-recovery-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-blocked-recovery-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-blocked-recovery-ad-"))

  try {
    const skills = ["Rust", "TypeScript", "React", "Python", "SQL"]
    for (let i = 0; i < CV_COUNT; i++) {
      writeFileSync(
        join(cvDir, `cv-${i + 1}.txt`),
        `Candidate ${i + 1} resume. Primary skills: ${skills[i]}, ${
          skills[(i + 2) % skills.length]
        }. Experience: ${i + 2} years. Previously at company ${i + 1}.`
      )
    }
    const adPath = join(adDir, "jobad.txt")
    writeFileSync(
      adPath,
      "Senior Platform Engineer. Required: Rust, TypeScript, React, Python, SQL."
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

    // Isolated real CLI: its own app dir (jobs/db) and model cache. Do NOT
    // download any embedder yet — the job must finish blocked.
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

    // Real job ad so the rubric has something to embed.
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

    // Committing the CV directory triggers processInWaves (SetupPane.commitCvDir).
    const input = page.locator('[data-testid="cv-dir-input"]')
    await input.click()
    await input.fill(cvDir)
    await input.press("Enter")

    // Candidates are scanned into the candidates table...
    await expect
      .poll(
        async () =>
          (
            (await invoke("list_candidates", {
              job_id: jobId,
              offset: 0,
              limit: 100,
            })) as { total: number }
          ).total,
        { timeout: 180000 }
      )
      .toBe(CV_COUNT)

    // ...and the cascade finishes blocked because no embedder is available.
    // After the fix the backend reports running = false for a blocked job.
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
        reason: expect.stringMatching(/^blocked:/),
      })

    // job_ui_state agrees: not running, ranking allowed, blocked reason present.
    const uiState = (await invoke("job_ui_state", {
      job_id: jobId,
    })) as {
      sort_allowed: boolean
      reason: string | null
      rank?: { running: boolean; converged: boolean }
    }
    expect(uiState.rank?.running).toBe(false)
    expect(uiState.sort_allowed).toBe(true)
    expect(uiState.reason).toMatch(/^blocked:/)

    // The Re-rank button is enabled: backend reports sort_allowed and not running.
    const rankControl = page.locator('[data-testid="rank-control"]')
    const rerankButton = rankControl.locator('[data-testid="rerank-button"]')
    await expect(rerankButton).toBeVisible({ timeout: 10000 })
    await expect(rerankButton).toBeEnabled()

    // Resolve the blocking condition: download the real embedders.
    await invoke("download_embedder", { slug: QUICK_MODEL })
    await invoke("download_embedder", { slug: DEEP_MODEL })

    // Click Re-rank. This is the crux of the fix: spawn_cascade must start a
    // NEW cascade over the blocked job instead of refusing it as "in flight".
    await rerankButton.click()

    // The blocked state is left behind: a new cascade is running, or (for a
    // fast run) it has already converged. Either way it is no longer blocked.
    await expect
      .poll(
        async () => {
          const st = (await invoke("get_rank_state", {
            job_id: jobId,
          })) as {
            running: boolean
            converged: boolean
          } | null
          return (st?.running ?? false) || (st?.converged ?? false)
        },
        { timeout: 120000 }
      )
      .toBe(true)

    // ...and the job converges once the embedder produces vectors.
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
        { timeout: 600000 }
      )
      .toMatchObject({
        running: false,
        converged: true,
        reason: "converged",
      })

    // Candidates now carry ranks and points.
    const ranked = (await invoke("list_candidates", {
      job_id: jobId,
      offset: 0,
      limit: 100,
    })) as {
      items: Array<{ rank: number | null; points: number | null }>
      total: number
    }
    const withScores = ranked.items.filter(
      (c) => c.rank != null && c.points != null
    ).length
    expect(withScores).toBe(CV_COUNT)
  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})