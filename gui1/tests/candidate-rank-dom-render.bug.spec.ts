import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression test for talentrankiq_gui/bug.md — "GUI Job Details Page — Candidates Never Show a Rank or Score"
//
// The bug: The Job Details page imports CVs, runs the ranking pipeline, and the
// backend really does rank every candidate. But the candidate table on the page
// keeps showing "—" in the Rank and Total columns indefinitely.
//
// Root cause: the candidate list only refetches when the total row count changes.
// Ranking does not change the row count, so once the table has loaded it never
// asks the backend again, and the rows it already rendered stay "—" forever.
//
// This test verifies the DOM renders ranks/points AFTER the backend has ranked,
// WITHOUT any user interaction (no sort/page/search/relike).

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 8

test("candidate table DOM shows rank and points after backend ranking completes (bug.md)", async ({
  page,
}) => {
  // Model downloads + first-time ORT session load dominate the runtime.
  test.setTimeout(30 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-dom-rank-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-dom-rank-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-dom-rank-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-dom-rank-ad-"))

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
      "Figma",
      "Leadership",
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

    // Isolated real CLI: its own app dir (jobs/db) and model cache.
    const cfg = await page.request.post("/api/test-config", {
      data: { appDir, modelDir },
    })
    expect(cfg.ok()).toBeTruthy()

    // Real downloads through the real CLI: the scout model powers the quick
    // stage; the default deep embedder is Harrier 270M ONNX (used by the
    // waves pipeline via runtime.model()).
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

    // Real job ad so the similarity targets are meaningful.
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

    // Committing the CV directory through the real GUI input triggers
    // processInWaves (SetupPane.commitCvDir).
    const input = page.locator('[data-testid="cv-dir-input"]')
    await input.click()
    await input.fill(cvDir)
    await input.press("Enter")

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

    // Candidates are scanned into the candidates table...
    await expect
      .poll(async () => (await listCandidates()).total, { timeout: 180000 })
      .toBe(CV_COUNT)

    // ...and the waves pipeline makes real deep-embedding progress
    // (get_rank_state folds the waves cv_embedding count for the runtime
    // model into embedded_done).
    await expect
      .poll(
        async () => {
          const st = (await invoke("get_rank_state", {
            job_id: jobId,
          })) as {
            embedded_done?: number
          } | null
          return st?.embedded_done ?? 0
        },
        { timeout: 600000 }
      )
      .toBeGreaterThan(0)

    // BUG VERIFICATION: The pipeline has produced embeddings and ranked candidates.
    // Now verify the DOM table cells show actual rank/points values (not "—")
    // WITHOUT any user interaction (no click, no sort, no search, no pagination).
    
    // Wait for the DOM to reflect the ranked data. The poll interval in the
    // GUI is 2.5s, so we need to wait long enough for at least one poll cycle
    // after ranking completes. We'll poll the DOM directly.
    await expect
      .poll(
        async () => {
          // Get all rank cells in the candidate table
          const rankCells = page.locator(
            '[data-testid="candidates-pane"] tbody tr td:first-child'
          )
          const count = await rankCells.count()
          if (count === 0) return 0
          
          const rankTexts = await rankCells.allTextContents()
          // Count cells that show actual numbers (not "—")
          return rankTexts.filter((text) => text.trim() !== "—").length
        },
        {
          timeout: 180000,
          message:
            "DOM rank cells never updated from '—' to actual numbers after backend ranking completed (bug.md)",
        }
      )
      .toBe(CV_COUNT)

    // Also verify the Total (points) column shows actual numbers
    await expect
      .poll(
        async () => {
          const totalCells = page.locator(
            '[data-testid="candidates-pane"] tbody tr td:nth-child(3)'
          )
          const count = await totalCells.count()
          if (count === 0) return 0
          
          const totalTexts = await totalCells.allTextContents()
          // Count cells that show actual numbers (not "—")
          return totalTexts.filter((text) => text.trim() !== "—").length
        },
        {
          timeout: 180000,
          message:
            "DOM total (points) cells never updated from '—' to actual numbers after backend ranking completed (bug.md)",
        }
      )
      .toBe(CV_COUNT)

    // Additional verification: the API also returns the correct data
    // (this should already pass based on the existing test, but let's confirm)
    await expect
      .poll(
        async () => {
          const page1 = await listCandidates()
          return page1.items.filter(
            (c) => c.rank != null && c.points != null
          ).length
        },
        {
          timeout: 120000,
          message:
            "candidates never received rank/points from process_in_waves (bug.md)",
        }
      )
      .toBe(CV_COUNT)
  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})