import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Chunk-level maxsim: a long CV with one strongly matching buried passage must
// outrank a short, topically similar CV, and the inspector's Match Lens
// (`chunk_match`) must show that matching passage with valid char ranges.
//
// No mocks: a real `triq serve`, real embedder downloads (the scout model plus
// the default deep Harrier 270M ONNX model), and the real ranking pipeline.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"

test("buried evidence outranks topical similarity and is shown by chunk_match", async ({
  page,
}) => {
  // Model downloads + first-time ORT session load dominate the runtime.
  test.setTimeout(25 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-chunk-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-chunk-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-chunk-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-chunk-ad-"))

  try {
    // A long CV whose only relevant evidence is one buried Rust passage.
    const filler =
      "I enjoy gardening, cooking, and traveling on weekends. ".repeat(400)
    const buriedCv = `${filler} Expert in Rust, Tokio, async runtimes, memory safety, and zero-cost abstractions.`
    // A short CV that is topically similar (backend/platform) but lacks Rust.
    const topicalCv =
      "Backend developer with Python, Java, SQL, Docker, and Kubernetes."
    writeFileSync(join(cvDir, "buried-rust.txt"), buriedCv)
    writeFileSync(join(cvDir, "topical-backend.txt"), topicalCv)

    const adPath = join(adDir, "jobad.txt")
    writeFileSync(
      adPath,
      "Rust systems engineer. Required: Rust, Tokio, async runtimes, memory safety, zero-cost abstractions."
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

    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

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
          id: string
          file_name: string
          rank: number | null
          points: number | null
          scores: Array<{
            question_id: string
            points: number
            max_points: number
            chunk_explanations: Array<{ text: string; similarity: number }>
          }>
        }>
        total: number
      }

    await expect
      .poll(async () => (await listCandidates()).total, { timeout: 180000 })
      .toBe(2)

    await expect
      .poll(
        async () => {
          const st = (await invoke("get_rank_state", {
            job_id: jobId,
          })) as { converged?: boolean } | null
          return st?.converged ?? false
        },
        { timeout: 600000 }
      )
      .toBe(true)

    const candidates = await listCandidates()
    const buried = candidates.items.find(
      (c) => c.file_name === "buried-rust.txt"
    )
    const topical = candidates.items.find(
      (c) => c.file_name === "topical-backend.txt"
    )
    expect(buried, "buried CV must be listed").toBeTruthy()
    expect(topical, "topical CV must be listed").toBeTruthy()

    // Buried evidence outranks topical similarity.
    expect(buried!.points).toBeGreaterThan(topical!.points!)
    expect(buried!.rank).toBeLessThan(topical!.rank!)

    // Every ad score carries a chunk explanation, and the buried CV's winning
    // chunk is the Rust passage.
    for (const c of candidates.items) {
      const ad = c.scores.find((s) => s.question_id === "ad")
      expect(ad, `${c.file_name} must have an ad score`).toBeTruthy()
      expect(
        ad!.chunk_explanations.length,
        `${c.file_name} ad score must explain its winning chunk`
      ).toBeGreaterThan(0)
      const best = ad!.chunk_explanations[0]
      expect(best.similarity).toBeCloseTo(ad!.points / ad!.max_points, 5)
    }
    const buriedAd = buried!.scores.find((s) => s.question_id === "ad")!
    expect(buriedAd.chunk_explanations[0].text.toLowerCase()).toContain("rust")

    // Match Lens: chunk_match returns the model and every chunk with char ranges
    // valid for the extracted text; the best chunk is the Rust passage.
    const detail = (await invoke("get_candidate", {
      candidate_id: buried!.id,
    })) as { extracted_text: string }
    const match = (await invoke("chunk_match", {
      candidate_id: buried!.id,
      target: "JobAd",
      chunk_size: 256,
    })) as {
      model: string
      chunks: Array<{ char_start: number; char_end: number; similarity: number }>
    }
    expect(match.model).toBe(DEEP_MODEL)
    expect(match.chunks.length).toBeGreaterThan(0)
    const text = detail.extracted_text
    for (const chunk of match.chunks) {
      expect(chunk.char_start).toBeGreaterThanOrEqual(0)
      expect(chunk.char_end).toBeGreaterThanOrEqual(chunk.char_start)
      expect(chunk.char_end).toBeLessThanOrEqual(text.length)
    }
    const bestChunk = match.chunks.reduce((a, b) =>
      b.similarity > a.similarity ? b : a
    )
    expect(
      text.slice(bestChunk.char_start, bestChunk.char_end).toLowerCase()
    ).toContain("rust")
  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})