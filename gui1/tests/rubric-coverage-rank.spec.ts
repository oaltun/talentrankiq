import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Rubric-chunk coverage scoring: a multi-requirement job ad is chunked into
// one query per requirement. Candidate A evidences every requirement in a
// separate passage; candidate B has one strong passage that covers only one
// requirement. Coverage scoring (mean over ad chunks of the best candidate
// cosine) must rank A above B, and the ad score fraction must equal the
// coverage score. The Match Lens (`chunk_match` for JobAd) must expose the
// matched ad chunk text/range for each CV chunk.
//
// No mocks: a real `triq serve`, real embedder downloads, and the real ranking
// pipeline.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"

test("multi-requirement coverage outranks a single-requirement candidate", async ({
  page,
}) => {
  test.setTimeout(25 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-rubric-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-rubric-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-rubric-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-rubric-ad-"))

  try {
    // A long ad whose requirements are separated so the tokenizer produces more
    // than one ad chunk, each dominated by one requirement.
    const rust =
      "Required: expert Rust programming, ownership, borrowing, memory safety, and async runtimes with Tokio. ".repeat(
        15
      )
    const k8s =
      "Required: Kubernetes, Docker, AWS, cloud infrastructure, and container orchestration at scale. ".repeat(
        15
      )
    const comm =
      "Required: excellent communication, mentorship, and technical leadership across distributed teams. ".repeat(
        15
      )
    const ad = `${rust}${k8s}${comm}`

    // A covers all three requirements in separate passages.
    const broadCv = `${"I am an expert Rust programmer with deep knowledge of ownership, borrowing, memory safety, and async runtimes using Tokio. ".repeat(
      15
    )}${"I have extensive experience with Kubernetes, Docker, AWS, cloud infrastructure, and container orchestration at scale. I communicate clearly, mentor engineers, and provide technical leadership across distributed teams. ".repeat(
      15
    )}`
    // B has one strong Rust passage and no Kubernetes or leadership evidence.
    const narrowCv =
      "I am an expert Rust programmer with deep knowledge of ownership, borrowing, memory safety, and async runtimes using Tokio. ".repeat(
        30
      )

    writeFileSync(join(cvDir, "broad-candidate.txt"), broadCv)
    writeFileSync(join(cvDir, "narrow-candidate.txt"), narrowCv)

    const adPath = join(adDir, "jobad.txt")
    writeFileSync(adPath, ad)

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
            chunk_explanations: Array<{
              text: string
              similarity: number
              rubric_text?: string | null
            }>
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
    const broad = candidates.items.find(
      (c) => c.file_name === "broad-candidate.txt"
    )
    const narrow = candidates.items.find(
      (c) => c.file_name === "narrow-candidate.txt"
    )
    expect(broad, "broad CV must be listed").toBeTruthy()
    expect(narrow, "narrow CV must be listed").toBeTruthy()

    const broadAd = broad!.scores.find((s) => s.question_id === "ad")!
    const narrowAd = narrow!.scores.find((s) => s.question_id === "ad")!
    expect(broadAd, "broad CV must have an ad score").toBeTruthy()
    expect(narrowAd, "narrow CV must have an ad score").toBeTruthy()

    // Breadth wins: the candidate covering every requirement outranks the
    // candidate whose one strong passage covers only one requirement.
    expect(broad!.points).toBeGreaterThan(narrow!.points!)
    expect(broad!.rank).toBeLessThan(narrow!.rank!)

    // The ad row point fraction equals the coverage score.
    expect(broadAd.points).toBeGreaterThan(0)
    const broadFraction = broadAd.points / broadAd.max_points
    expect(broadFraction).toBeGreaterThan(0)
    expect(broadFraction).toBeLessThanOrEqual(1)

    // The explanation names the matched requirement (rubric chunk).
    expect(
      broadAd.chunk_explanations.length,
      "ad score must explain its winning chunk"
    ).toBeGreaterThan(0)
    const best = broadAd.chunk_explanations[0]
    expect(best.similarity).toBeCloseTo(broadFraction, 5)
    expect(best.rubric_text, "explanation must name the matched rubric chunk")
      .toBeTruthy()
    expect(best.rubric_text!.toLowerCase()).toContain("required")

    // Match Lens: chunk_match for JobAd returns each CV chunk's best-over-ad
    // cosine plus the matched ad chunk's text and char range (valid for the ad
    // text).
    const detail = (await invoke("get_candidate", {
      candidate_id: broad!.id,
    })) as { extracted_text: string }
    const adText = (await invoke("get_job_ad", {
      job_id: jobId,
    })) as { text: string }
    const match = (await invoke("chunk_match", {
      candidate_id: broad!.id,
      target: "JobAd",
      chunk_size: 256,
    })) as {
      model: string
      chunks: Array<{
        char_start: number
        char_end: number
        similarity: number
        matched_target_text?: string | null
        matched_target_char_start?: number | null
        matched_target_char_end?: number | null
      }>
    }
    expect(match.model).toBe(DEEP_MODEL)
    expect(match.chunks.length).toBeGreaterThan(0)

    const cvText = detail.extracted_text
    for (const chunk of match.chunks) {
      // CV chunk range is valid for the extracted text.
      expect(chunk.char_start).toBeGreaterThanOrEqual(0)
      expect(chunk.char_end).toBeGreaterThanOrEqual(chunk.char_start)
      expect(chunk.char_end).toBeLessThanOrEqual(cvText.length)
      // JobAd similarity comes with a matched ad chunk whose range is valid for
      // the ad text.
      expect(
        chunk.matched_target_text,
        "JobAd chunk_match must name the matched ad chunk"
      ).toBeTruthy()
      expect(chunk.matched_target_char_start).toBeGreaterThanOrEqual(0)
      expect(chunk.matched_target_char_end).toBeGreaterThanOrEqual(
        chunk.matched_target_char_start!
      )
      expect(chunk.matched_target_char_end).toBeLessThanOrEqual(adText.text.length)
      const matchedSlice = adText.text.slice(
        chunk.matched_target_char_start!,
        chunk.matched_target_char_end!
      )
      expect(matchedSlice).toBe(chunk.matched_target_text)
    }

    // The best CV chunk's similarity is its best cosine over ad chunks.
    const bestChunk = match.chunks.reduce((a, b) =>
      b.similarity > a.similarity ? b : a
    )
    expect(bestChunk.similarity).toBeGreaterThan(0)
    expect(bestChunk.matched_target_text!.toLowerCase()).toContain("required")
  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})