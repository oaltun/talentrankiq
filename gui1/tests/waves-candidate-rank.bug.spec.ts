import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression test for talentrankiq_gui/bug.md — "GUI Job Details Page":
//
//   The Job Details page imports CVs and runs the ranking pipeline, but the
//   candidates never get a rank or a total score: the table shows "—" in the
//   Rank and Total columns forever, even while the backend reports that
//   ranking is running and deep embeddings are being produced.
//
// Root cause: the GUI's only ranking trigger is process_in_waves, but that
// pipeline writes results to its own cv / cv_embedding / cv_similarity /
// cv_comparison / waves_rank_state tables and never writes back to the
// candidates / candidate_scores tables that list_candidates — and therefore
// the GUI table — reads.
//
// No mocks: a real `triq serve`, real embedder downloads (the scout model
// plus the default deep Harrier 270M ONNX model), and the real waves pipeline
// started exactly the way the GUI starts it (committing the CV directory in
// the workspace auto-triggers processInWaves).

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 12

test("process_in_waves gives candidates a rank and points (bug.md)", async ({
  page,
}) => {
  // Model downloads + first-time ORT session load dominate the runtime.
  test.setTimeout(20 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-waves-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-waves-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-waves-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-waves-ad-"))

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
          const st = (await invoke("get_rank_state", { job_id: jobId })) as {
            embedded_done?: number
          } | null
          return st?.embedded_done ?? 0
        },
        { timeout: 600000 }
      )
      .toBeGreaterThan(0)

    // BUG: the pipeline has produced embeddings, yet list_candidates — the
    // source the GUI table renders — still reports rank/points NULL for every
    // candidate, so the page shows "—" forever.
    await expect
      .poll(
        async () => {
          const page1 = await listCandidates()
          return page1.items.filter((c) => c.rank != null && c.points != null)
            .length
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