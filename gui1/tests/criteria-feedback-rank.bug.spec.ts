import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression test for talentrankiq_gui/bug.md — "GUI Job Details Page — Ranking Ignores Criteria and Feedback":
//
// Bug 1: Feedback (like) does not affect the ranking.
//   - Clicking Like changes the star, but the ranking/total does not move.
//   - Root cause: feedback rows are never written to candidate_scores, so rescore
//     has no feedback grades to read.
//
// Bug 2: Criteria added AFTER waves run may not be scored in the next write-back.
//   - The waves pipeline's write_back_candidate_scores should pivot every target
//     hash (job ad + each criterion), but may only persist the job-ad row.
//
// No mocks: real `triq serve`, real embedder downloads, real waves pipeline.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"

test.describe.serial("bug.md: feedback and criteria affect ranking", () => {
  test("like/feedback changes the ranking and adds fb_job score row", async ({
    page,
  }) => {
    test.setTimeout(20 * 60 * 1000)

    const appDir = mkdtempSync(join(tmpdir(), "triq-fb-app-"))
    const modelDir = mkdtempSync(join(tmpdir(), "triq-fb-models-"))
    const cvDir = mkdtempSync(join(tmpdir(), "triq-fb-cvs-"))
    const adDir = mkdtempSync(join(tmpdir(), "triq-fb-ad-"))

    try {
      // Create CVs where one is clearly better for the job ad but we'll like the other.
      const cvs = [
        { file: "cv-a.txt", text: "Senior Platform Engineer. 5 years Kubernetes, Go, AWS, Docker. Managed large clusters." },
        { file: "cv-b.txt", text: "Backend Developer. Go, Python, SQL. 3 years experience. Some Docker." },
        { file: "cv-c.txt", text: "Platform Engineer. Kubernetes, Helm, AWS. 2 years." },
      ]
      for (const cv of cvs) {
        writeFileSync(join(cvDir, cv.file), cv.text)
      }

      const adPath = join(adDir, "jobad.txt")
      writeFileSync(
        adPath,
        "Senior Platform Engineer. Required: Kubernetes, Go, AWS, Docker."
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
      expect(jobId).toBeTruthy()

      await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

      // Add a criterion to make the rubric more interesting.
      await invoke("add_criterion", {
        job_id: jobId,
        text: "Kubernetes",
        weight: 5.0,
      })

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
            points_max: number | null
            liked: boolean
            scores: Array<{ question_id: string; label: string; points: number; max_points: number }>
          }>
          total: number
        }

      await expect
        .poll(async () => (await listCandidates()).total, { timeout: 180000 })
        .toBe(3)

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

      // Get initial ranking.
      const initial = await listCandidates()
      const cvA = initial.items.find((c) => c.file_name === "cv-a.txt")!
      const cvB = initial.items.find((c) => c.file_name === "cv-b.txt")!
      const cvC = initial.items.find((c) => c.file_name === "cv-c.txt")!

      // Verify initial state: all candidates have scores, points_max is rubric capacity (70).
      expect(initial.items.length).toBe(3)
      for (const c of initial.items) {
        expect(c.points_max).toBe(70.0)
        const qids = c.scores.map((s) => s.question_id)
        expect(qids).toContain("ad")
        expect(qids.some((id) => id.startsWith("crit_"))).toBe(true)
      }

      // Like cv-b (a candidate) - this should boost its feedback score.
      await invoke("apply_feedback", {
        job_id: jobId,
        ordered_candidate_ids: initial.items.map((c) => c.id),
        liked_candidate_ids: [cvB.id],
      })

      // After feedback, re-fetch candidates. The ranking should have changed:
      // cv-b should move up because of the feedback signal.
       const afterFeedback = await listCandidates()
       console.log("AFTER:", afterFeedback.items.map(c => ({file: c.file_name, points: c.points, rank: c.rank})))
       const cvBAfter = afterFeedback.items.find((c) => c.file_name === "cv-b.txt")!
      const cvBBefore = initial.items.find((c) => c.file_name === "cv-b.txt")!

      // BUG: Currently, liking a candidate changes the star but the ranking/total
      // does not move because feedback rows are never written to candidate_scores.
      expect(cvBAfter.points).toBeGreaterThan(cvBBefore.points)
      expect(cvBAfter.rank).toBeLessThan(cvBBefore.rank) // Rank improves (lower number = better)

      // Verify the feedback score row exists in candidate_scores.
      const cvBScores = cvBAfter.scores
      const fbJobScore = cvBScores.find((s) => s.question_id === "fb_job")
      expect(fbJobScore).toBeDefined()
      expect(fbJobScore!.points).toBeGreaterThan(0)

      // Reset feedback: the learned fb_job vector must be removed, the liked
      // candidate's fb_job points must fall to zero, totals/rank are recomputed,
      // and no candidate remains liked.
      await invoke("reset_feedback", { job_id: jobId })

      const afterReset = await listCandidates()
      const cvBAfterReset = afterReset.items.find((c) => c.file_name === "cv-b.txt")!
      expect(cvBAfterReset).toBeDefined()

      const fbJobAfterReset = cvBAfterReset.scores.find(
        (s) => s.question_id === "fb_job"
      )
      expect(fbJobAfterReset).toBeDefined()
      expect(fbJobAfterReset!.points).toBe(0)

      // The feedback contribution is gone, so the total drops back below the
      // liked value and rank is recomputed.
      expect(cvBAfterReset.points).toBeLessThan(cvBAfter.points)
      expect(cvBAfterReset.rank).not.toBeNull()

      const likedAfterReset = (await invoke("list_liked_candidate_ids", {
        job_id: jobId,
      })) as string[]
      expect(likedAfterReset.length).toBe(0)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(cvDir, { recursive: true, force: true })
      rmSync(adDir, { recursive: true, force: true })
    }
  })

  test("criterion added AFTER waves run is scored in next write-back", async ({
    page,
  }) => {
    test.setTimeout(20 * 60 * 1000)

    const appDir = mkdtempSync(join(tmpdir(), "triq-crit-after-app-"))
    const modelDir = mkdtempSync(join(tmpdir(), "triq-crit-after-models-"))
    const cvDir = mkdtempSync(join(tmpdir(), "triq-crit-after-cvs-"))
    const adDir = mkdtempSync(join(tmpdir(), "triq-crit-after-ad-"))

    try {
      // Create CVs with varying relevance to "Kubernetes" criterion
      const cvs = [
        { file: "cv-1.txt", text: "Senior DevOps Engineer. 5 years Kubernetes, Docker, AWS. Managed 50-node clusters." },
        { file: "cv-2.txt", text: "Backend Developer. Go, Python, SQL. Some Docker experience. No Kubernetes." },
        { file: "cv-3.txt", text: "Platform Engineer. Kubernetes expert, 3 years. Helm, ArgoCD, Prometheus." },
        { file: "cv-4.txt", text: "Full Stack Developer. React, TypeScript, Node.js. No container orchestration." },
      ]
      for (const cv of cvs) {
        writeFileSync(join(cvDir, cv.file), cv.text)
      }

      const adPath = join(adDir, "jobad.txt")
      writeFileSync(
        adPath,
        "Senior Platform Engineer. Required: Kubernetes, Go, AWS, Docker."
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
      expect(jobId).toBeTruthy()

      await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

      // FIRST: Import CVs and let waves run WITHOUT any criterion.
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
            points_max: number | null
            scores: Array<{ question_id: string; label: string; points: number; max_points: number }>
          }>
          total: number
        }

      await expect
        .poll(async () => (await listCandidates()).total, { timeout: 180000 })
        .toBe(4)

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

      // Wait for waves to converge.
      await expect
        .poll(
          async () => {
            const st = (await invoke("get_rank_state", { job_id: jobId })) as {
              converged?: number
            } | null
            return st?.converged ?? 0
          },
          { timeout: 600000 }
        )
        .toBe(1)

      // At this point, waves has run and written back scores. Without a criterion,
      // points_max should be 65 (ad 50 + fb_job 10 + fb_profile 5 = 65).
      const beforeCriterion = await listCandidates()
      console.log("Before criterion:", beforeCriterion.items.map(c => ({file: c.file_name, points: c.points, points_max: c.points_max, scores: c.scores.map(s => s.question_id)})))

      // SECOND: Add a criterion AFTER waves has run.
      await invoke("add_criterion", {
        job_id: jobId,
        text: "Kubernetes",
        weight: 5.0,
      })

      // The add_criterion API calls inputs_changed which triggers a rescore.
      // Wait a bit for the rescore to complete.
      await page.waitForTimeout(2000)

      const afterCriterion = await listCandidates()
      console.log("After criterion:", afterCriterion.items.map(c => ({file: c.file_name, points: c.points, points_max: c.points_max, scores: c.scores.map(s => s.question_id)})))

      // The criterion should now be scored: candidate_scores should have the criterion row
      // and points_max should be 70 (ad 50 + fb_job 10 + fb_profile 5 + criterion 5 = 70).
      for (const candidate of afterCriterion.items) {
        const questionIds = candidate.scores.map((s) => s.question_id)
        // Should have 'ad' AND the criterion id
        expect(questionIds).toContain("ad")
        expect(questionIds.some((id) => id.startsWith("crit_"))).toBe(true)
      }

      // Verify points_max equals the job's rubric capacity (ad 50 + fb_job 10 + fb_profile 5 + criterion 5 = 70).
      for (const candidate of afterCriterion.items) {
        expect(candidate.points_max).toBe(70.0)
      }

      // Verify the ranking reflects the criterion: candidates with Kubernetes
      // experience should rank higher than those without.
      const cv1 = afterCriterion.items.find((c) => c.file_name === "cv-1.txt")!
      const cv2 = afterCriterion.items.find((c) => c.file_name === "cv-2.txt")!
      const cv3 = afterCriterion.items.find((c) => c.file_name === "cv-3.txt")!
      const cv4 = afterCriterion.items.find((c) => c.file_name === "cv-4.txt")!

      expect(cv1.points).toBeGreaterThan(cv2.points)
      expect(cv3.points).toBeGreaterThan(cv4.points)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(cvDir, { recursive: true, force: true })
      rmSync(adDir, { recursive: true, force: true })
    }
  })
})