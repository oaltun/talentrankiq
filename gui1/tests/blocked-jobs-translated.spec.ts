import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Test: Jobs with no ad and jobs with no usable embedder show translated
// blocked reasons with converged: false and no silent empty rank rows.

const QUICK_MODEL = "static-similarity-mrl-multilingual-v1"
const DEEP_MODEL = "harrier-oss-v1-270m-ONNX-q4f16"
const CV_COUNT = 4

test("job with no ad shows translated blocked reason", async ({ page }) => {
  test.setTimeout(15 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-no-ad-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-no-ad-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-no-ad-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-no-ad-ad-"))

  try {
    for (let i = 0; i < CV_COUNT; i++) {
      writeFileSync(
        join(cvDir, `cv-${i + 1}.txt`),
        `Candidate ${i + 1} resume. Skills: Rust, TypeScript. Experience: ${i + 2} years.`
      )
    }
    // No job ad file created

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

    // Commit CV directory WITHOUT setting a job ad
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

    // Verify blocked state via get_rank_state
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

    // Verify workspace shows translated blocked reason
    const rankControl = page.locator('[data-testid="rank-control"]')
    await expect(rankControl).toBeVisible({ timeout: 10000 })

    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })

    // Poll for the translated blocked reason to appear in the GUI
    await expect
      .poll(
        async () => {
          const text = await statusLabel.textContent()
          return text?.toLowerCase() ?? ""
        },
        { timeout: 30000 }
      )
      .toMatch(/no job ad|iş ilanı ayarlanmadı/)

    const statusText = await statusLabel.textContent()
    expect(statusText).toBeTruthy()
    expect(statusText).not.toMatch(/^blocked:/)
    // Should show translated "No job ad set" or Turkish equivalent
    expect(statusText?.toLowerCase()).toMatch(/no job ad|iş ilanı ayarlanmadı/)

    // Verify data-reason attribute
    const dataReason = await statusLabel.getAttribute("data-reason")
    expect(dataReason).toBe("no-jobad")

    // Verify no silent empty rank rows — candidates should have rank/points as null
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

    const candidates = await listCandidates()
    for (const c of candidates.items) {
      expect(c.rank).toBeNull()
      expect(c.points).toBeNull()
    }

    // Verify Re-rank button is disabled with translated reason
    const rerankButton = rankControl.locator('[data-testid="rerank-button"]')
    await expect(rerankButton).toBeDisabled()

    const disabledReason = rankControl.locator('[data-testid="rerank-disabled-reason"]')
    const reasonText = await disabledReason.textContent()
    expect(reasonText).toBeTruthy()
    expect(reasonText).not.toMatch(/^blocked:/)
    expect(reasonText?.toLowerCase()).toMatch(/no job ad|iş ilanı ayarlanmadı/)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})

test("job with no usable embedder shows translated blocked reason", async ({
  page,
}) => {
  test.setTimeout(15 * 60 * 1000)

  const appDir = mkdtempSync(join(tmpdir(), "triq-no-embedder-app-"))
  const modelDir = mkdtempSync(join(tmpdir(), "triq-no-embedder-models-"))
  const cvDir = mkdtempSync(join(tmpdir(), "triq-no-embedder-cvs-"))
  const adDir = mkdtempSync(join(tmpdir(), "triq-no-embedder-ad-"))

  try {
    for (let i = 0; i < CV_COUNT; i++) {
      writeFileSync(
        join(cvDir, `cv-${i + 1}.txt`),
        `Candidate ${i + 1} resume. Skills: Rust, TypeScript. Experience: ${i + 2} years.`
      )
    }
    const adPath = join(adDir, "jobad.txt")
    writeFileSync(
      adPath,
      "Senior Engineer. Required: Rust, TypeScript."
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

    // Do NOT download any embedder

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

    // Set job ad
    await invoke("set_job_ad_path", { job_id: jobId, path: adPath })

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
        reason: expect.stringMatching(/^blocked:no-active-embedder$/),
      })

    // Verify workspace shows translated blocked reason
    const rankControl = page.locator('[data-testid="rank-control"]')
    await expect(rankControl).toBeVisible({ timeout: 10000 })

    const statusLabel = rankControl.locator('[data-testid="rank-status-label"]')
    await expect(statusLabel).toBeVisible({ timeout: 10000 })

    // Poll for the translated blocked reason to appear in the GUI
    await expect
      .poll(
        async () => {
          const text = await statusLabel.textContent()
          return text?.toLowerCase() ?? ""
        },
        { timeout: 30000 }
      )
      .toMatch(/no active embedder|aktif gömme modeli yok/)

    const statusText = await statusLabel.textContent()
    expect(statusText).toBeTruthy()
    expect(statusText).not.toMatch(/^blocked:/)
    // Should show translated "No active embedder" or Turkish equivalent
    expect(statusText?.toLowerCase()).toMatch(/no active embedder|aktif gömme modeli yok/)

    // Verify data-reason attribute
    const dataReason = await statusLabel.getAttribute("data-reason")
    expect(dataReason).toBe("no-active-embedder")

    // Verify no silent empty rank rows
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

    const candidates = await listCandidates()
    for (const c of candidates.items) {
      expect(c.rank).toBeNull()
      expect(c.points).toBeNull()
    }

    // Verify Re-rank button is disabled with translated reason
    const rerankButton = rankControl.locator('[data-testid="rerank-button"]')
    await expect(rerankButton).toBeDisabled()

    const disabledReason = rankControl.locator('[data-testid="rerank-disabled-reason"]')
    const reasonText = await disabledReason.textContent()
    expect(reasonText).toBeTruthy()
    expect(reasonText).not.toMatch(/^blocked:/)
    expect(reasonText?.toLowerCase()).toMatch(/no active embedder|aktif gömme modeli yok/)

    // Test language switching updates labels
    // Open settings and switch to Turkish
    await page.locator('[data-testid="sidebar-secondary-settings-btn"]').click()
    await expect(page.locator('[data-testid="settings-dialog"]')).toBeVisible()
    
    // Find and click Turkish language option
    await page.locator('[data-testid="language-option-tr"]').click()
    // Close dialog by clicking outside or pressing Escape
    await page.keyboard.press("Escape")
    await expect(page.locator('[data-testid="settings-dialog"]')).toBeHidden()

    // Wait for language to switch and verify Turkish labels
    await page.waitForTimeout(1000)
    
    const statusTextTr = await statusLabel.textContent()
    expect(statusTextTr).toBeTruthy()
    expect(statusTextTr).not.toMatch(/^blocked:/)
    expect(statusTextTr).toMatch(/aktif gömme modeli yok|iş ilanı ayarlanmadı/)

    const disabledReasonTr = rankControl.locator('[data-testid="rerank-disabled-reason"]')
    const reasonTextTr = await disabledReasonTr.textContent()
    expect(reasonTextTr).toBeTruthy()
    expect(reasonTextTr).not.toMatch(/^blocked:/)
    expect(reasonTextTr).toMatch(/aktif gömme modeli yok|iş ilanı ayarlanmadı/)

  } finally {
    rmSync(appDir, { recursive: true, force: true })
    rmSync(modelDir, { recursive: true, force: true })
    rmSync(cvDir, { recursive: true, force: true })
    rmSync(adDir, { recursive: true, force: true })
  }
})