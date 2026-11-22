import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync, appendFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test: New CV added to watched directory after convergence triggers re-rank.
// Assert the new candidate is extracted, embedded, scored, and the status
// passes through running before returning to converged.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const INITIAL_CV_COUNT = 4
const TOTAL_CV_COUNT = 5

test("new CV after convergence triggers re-rank", async ({ page }) => {
  test.setTimeout(25 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-new-cv-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-new-cv-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-new-cv-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-new-cv-ad-"))

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
    // Create initial CVs
    for (let i = 0; i < INITIAL_CV_COUNT; i++) {
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

    // Set job ad
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

    // Commit CV directory
    const input = page.locator('[data-testid="cv-dir-input"]')
    await input.click()
    await input.fill(cvDir)
    await input.press("Enter")

    // Wait for initial candidates to be scanned
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
      .toBe(INITIAL_CV_COUNT)

    // Wait for convergence
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

    // Verify initial candidates have ranks and points
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
            "initial candidates never received non-zero rank/points",
        }
      )
      .toBe(INITIAL_CV_COUNT)

    // Verify workspace shows converged
    const rankControl = page.locator('[data-testid="rank-control"]')
    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })
    let statusText = await statusLabel.textContent()
    expect(statusText?.toLowerCase()).toMatch(/converged|yakındır/)

    // NOW add a new CV to the watched directory
    const newCvPath = join(cvDir, `cv-${INITIAL_CV_COUNT + 1}.txt`)
    writeFileSync(
      newCvPath,
      `Candidate ${INITIAL_CV_COUNT + 1} resume. Primary skills: ${skills[INITIAL_CV_COUNT]}, ${skills[(INITIAL_CV_COUNT + 3) % skills.length]}. Experience: ${INITIAL_CV_COUNT + 2} years. Previously at company ${INITIAL_CV_COUNT + 1}.`
    )

    // The file watcher should detect the new CV and trigger a re-rank
    // Wait for the new candidate to appear (total should be 5)
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
      .toBe(TOTAL_CV_COUNT)

    // During the re-rank, the status should show running
    // Poll for running status
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
          return st?.running ?? false
        },
        { timeout: 60000, intervals: [50, 50, 50, 100, 100] }
      )
      .toBe(true)

    // Then wait for convergence again
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

    // Verify all candidates (including new one) have ranks and points
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
            "new candidate never received non-zero rank/points",
        }
      )
      .toBe(TOTAL_CV_COUNT)

    // Verify workspace shows converged again
    statusText = await statusLabel.textContent()
    expect(statusText?.toLowerCase()).toMatch(/converged|yakındır/)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})