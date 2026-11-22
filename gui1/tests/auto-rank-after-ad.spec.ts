import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test: Automatic rank after set_job_ad_path runs after CV import.
// The job should rank with no further user action.
// Assert non-zero points, a converged rank state, and a converged workspace status.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 5

test("automatic rank after ad set: job ranks with no manual trigger", async ({
  page,
}) => {
  test.setTimeout(20 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-auto-rank-ad-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-auto-rank-ad-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-auto-rank-ad-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-auto-rank-ad-ad-"))

  try {
    const skills = [
      "Rust",
      "TypeScript",
      "React",
      "Python",
      "SQL",
      "Kubernetes",
      "Go",
      "GraphQL",
      "Docker",
      "AWS",
    ]
    for (let i = 0; i < CV_COUNT; i++) {
      writeFileSync(
        join(cvDir, `cv-${i + 1}.txt`),
        `Candidate ${i + 1} resume. Primary skills: ${skills[i]}, ${
          skills[(i + 3) % skills.length]
        }. Experience: ${i + 2} years. Previously at company ${i + 1}.`
      )
    }
    const adPath = join(adDir, "jobad.txt")
    writeFileSync(
      adPath,
      "Senior Platform Engineer. Required: Kubernetes, Go, AWS, Docker, SQL. Nice to have: Rust, GraphQL."
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

    const cfg = await page.request.post("/api/test-config", {
      data: { appDir, modelDir },
    })
    expect(cfg.ok()).toBeTruthy()

    await invoke("download_embedder", { slug: QUICK_MODEL })
    await invoke("download_embedder", { slug: DEEP_MODEL })

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

    // First, commit the CV directory WITHOUT a job ad
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

    // At this point, the job should be blocked (no job ad)
    // Verify blocked state
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
        reason: expect.stringMatching(/^blocked:no-jobad$/),
      })

    // Now set the job ad path — this should trigger an automatic re-rank
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

    // Wait for the ranking to complete and converge
    await expect
      .poll(
        async () => {
          const st = (await invoke("get_rank_state", {
            job_id: jobId,
          })) as {
            running: boolean
            converged: boolean
            reason: string | null
            candidates: number
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

    // Verify candidates have non-zero points and ranks
    const listCandidates = async () =>
      (await invoke("list_candidates", {
        job_id: jobId,
        offset: 0,
        limit: 100,
      })) as {
        items: Array<{
          file_name: string
          rank: number | null
          points: number | null
        }>
        total: number
      }

    await expect
      .poll(
        async () => {
          const page1 = await listCandidates()
          return page1.items.filter(
            (c) => c.rank != null && c.points != null && c.points > 0
          ).length
        },
        {
          timeout: 180000,
          message:
            "candidates never received non-zero rank/points after ad set",
        }
      )
      .toBe(CV_COUNT)

    // Verify the workspace shows converged status
    const rankControl = page.locator('[data-testid="rank-control"]')
    await expect(rankControl).toBeVisible({ timeout: 10000 })

    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })

    const statusText = await statusLabel.textContent()
    expect(statusText).toBeTruthy()
    expect(statusText).not.toMatch(/^blocked:/)
    expect(statusText).not.toMatch(/stale/i)

    // The converged label should be shown (localized)
    expect(statusText?.toLowerCase()).toMatch(/converged|yakındır/)

    // Verify the data-reason is empty or not a blocked code
    const dataReason = await statusLabel.getAttribute("data-reason")
    expect(dataReason).not.toMatch(/^blocked:/)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})