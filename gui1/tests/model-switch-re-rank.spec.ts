import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test: Switching the default embedding model on a converged job re-ranks
// every candidate in the new space and reports converged only after coverage,
// never before.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL_1 = "harrier-oss-v1-270m-ONNX-q4f16"
// Use the MLX variant as the second model (same size, different backend)
// which is in the registry and available on macOS.
const DEEP_MODEL_2 = "harrier-oss-v1-270m-MLX-4bit"
const CV_COUNT = 4

test("model switch on converged job re-ranks before converged", async ({
  page,
}) => {
  test.setTimeout(30 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-model-switch-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-model-switch-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-model-switch-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-model-switch-ad-"))

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

    // Download both deep models
    await invoke("download_embedder", { slug: QUICK_MODEL })
    await invoke("download_embedder", { slug: DEEP_MODEL_1 })
    await invoke("download_embedder", { slug: DEEP_MODEL_2 })

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

    // Set default model to DEEP_MODEL_1 first
    await invoke("make_default_embedder", { slug: DEEP_MODEL_1 })

    // Commit CV directory
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

    // Wait for convergence with first model
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

    // Capture the points from the first model for comparison
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

    const firstModelPoints = await listCandidates()
    const pointsMap1 = new Map<string, number>()
    for (const c of firstModelPoints.items) {
      if (c.points != null) pointsMap1.set(c.file_name, c.points)
    }

    // Verify workspace shows converged
    const rankControl = page.locator('[data-testid="rank-control"]')
    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })
    let statusText = await statusLabel.textContent()
    expect(statusText?.toLowerCase()).toMatch(/converged|yakındır/)

    // NOW switch to the second deep model
    await invoke("make_default_embedder", { slug: DEEP_MODEL_2 })

    // The model switch should trigger a re-rank
    // First, status should show running
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

    // During re-rank, converged should be false
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
          return st?.converged ?? true
        },
        { timeout: 60000 }
      )
      .toBe(false)

    // Then wait for convergence with the new model
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

    // Verify all candidates have been re-scored in the new space
    const secondModelPoints = await listCandidates()
    const pointsMap2 = new Map<string, number>()
    for (const c of secondModelPoints.items) {
      if (c.points != null) pointsMap2.set(c.file_name, c.points)
    }

    // Points should have changed for at least some candidates (different embedding space)
    let pointsChanged = false
    for (const [name, points1] of pointsMap1) {
      const points2 = pointsMap2.get(name)
      if (points2 != null && points2 !== points1) {
        pointsChanged = true
        break
      }
    }
    expect(pointsChanged).toBe(true)

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