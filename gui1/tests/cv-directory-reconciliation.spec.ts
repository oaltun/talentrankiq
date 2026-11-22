import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync, renameSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test: Editing, deleting, and renaming CVs in the watched directory triggers
// reconciliation on every ScanDirectory scan. Real files, real models, real
// CLI — no mocks.
//
// Scenarios:
//   - modified file -> candidate reset and re-imported
//   - deleted file -> candidate retired, leaves list_candidates
//   - renamed file -> old row retired, exactly one new active candidate
//   - all three outcomes converge in a single scan
//
// Asserts the existing suites stay green by leaving other tests untouched.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const INITIAL_CV_COUNT = 3

test("edit, delete, and rename CVs reconcile in a single scan", async ({
  page,
}) => {
  test.setTimeout(30 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-reconcile-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-reconcile-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-reconcile-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-reconcile-ad-"))

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
    // Create initial CVs.
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

    // Commit CV directory.
    const input = page.locator('[data-testid="cv-dir-input"]')
    await input.click()
    await input.fill(cvDir)
    await input.press("Enter")

    // Wait for initial candidates to be scanned.
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

    // Wait for convergence.
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

    // Initial candidates have ranks and points.
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
          message: "initial candidates never received non-zero rank/points",
        }
      )
      .toBe(INITIAL_CV_COUNT)

    // -- Reconciliation: edit, delete, rename in one pass --
    const cv2 = join(cvDir, "cv-2.txt")
    writeFileSync(
      cv2,
      `Candidate 2 resume. Primary skills: ${skills[1]}, ${
        skills[4]
      }. Experience: ${2 + 5} years. Previously at company ${2 + 10}.`
    ) // modified content

    rmSync(join(cvDir, "cv-3.txt")) // deleted

    renameSync(join(cvDir, "cv-1.txt"), join(cvDir, "cv-1-renamed.txt")) // renamed

    // All three outcomes converge in one scan:
    //   - cv-1 -> retired (deleted) + cv-1-renamed -> new active
    //   - cv-2 -> reset and re-imported
    //   - cv-3 -> retired (deleted)
    // Final total: cv-1-renamed.txt + cv-2.txt = 2 active candidates.
    await expect
      .poll(
        async () => {
          const result = await listCandidates()
          return result.total
        },
        { timeout: 300000 }
      )
      .toBe(2)

    // Wait for re-convergence.
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

    const final = await listCandidates()
    const names = final.items.map((c) => c.file_name).sort()
    expect(names).toEqual(["cv-1-renamed.txt", "cv-2.txt"])

    // The deleted file is gone from listings.
    expect(final.items.find((c) => c.file_name === "cv-3.txt")).toBeUndefined()

    // Remaining candidates were re-ranked.
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
          message: "reconciled candidates never received non-zero rank/points",
        }
      )
      .toBe(2)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})
