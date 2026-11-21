// note to dev: use only stable ids, not css. if gui does not have id, add it.
import { test, expect, Page } from "@playwright/test"
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const HERE = dirname(fileURLToPath(import.meta.url))
const APP_DIR = () => mkdtempSync(join(tmpdir(), "triq-flow-app-"))
const MODEL_DIR = () => mkdtempSync(join(tmpdir(), "triq-flow-models-"))
const CV_DIR = () => mkdtempSync(join(tmpdir(), "triq-flow-cvs-"))

async function useIsolatedCli(page: Page, appDir: string, modelDir: string) {
  const res = await page.request.post("/api/test-config", {
    data: { appDir, modelDir },
  })
  expect(res.ok()).toBeTruthy()
}

test.describe("App Shell", () => {
  test("navigates to Models activity", async ({ page }) => {
    test.setTimeout(600000)

    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    const cvDir = CV_DIR()

    try {
      // Point the dev proxy at an isolated real `triq serve` BEFORE loading the
      // app, so this test never shares CLI state with (or inherits a still-busy
      // child from) whichever spec ran before it.
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      // CV WATCH TEST - manual pipeline controls removed
      await page.locator('[data-testid="activity-home"]').click()

      // Add a new job
      await page.locator('[data-testid="add-job-button"]').click()
      await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
        timeout: 10000,
      })

      const jobList = page.locator('[data-testid="job-list"]')
      const jobItems = jobList.locator('[data-testid^="job-item-"]')
      expect(await jobItems.count()).toBeGreaterThanOrEqual(1)

      // Open job workspace - must render directly even though the quick model
      // is missing (regression: bug.md). No redirect to Models, no infobar.
      await jobItems.first().click()
      const workspacePage = page.locator('[data-testid="workspace-page"]')
      await expect(workspacePage).toBeVisible({ timeout: 30000 })
      await expect(page.locator('[data-testid="infobar"]')).toHaveCount(0)
      await expect(page).not.toHaveURL(/#\/models\?.*reason=quick/)

      // Manual pipeline controls removed
      await expect(page.locator('[data-testid="convert-btn"]')).toHaveCount(0)
      await expect(page.locator('[data-testid="quick-embed-btn"]')).toHaveCount(0)
      await expect(page.locator('[data-testid="deep-embed-btn"]')).toHaveCount(0)
      await expect(page.locator('[data-testid="watch-cv-dir-switch"]')).toHaveCount(0)
      await expect(page.locator('[data-testid="run-ranking-btn"]')).toHaveCount(0)

      // Change CV watch directory
      const testCvDir = "/Users/apple1/Documents/repo/act/ik/activeik/talentrankiq_private/test-cv/Job10CVs/cv"
      const cvDirInput = page.locator('[data-testid="cv-dir-input"]')
      await cvDirInput.click()
      await cvDirInput.fill(testCvDir)
      await expect(cvDirInput).toHaveValue(testCvDir)
      await cvDirInput.press("Enter")

      // Candidates should appear
      const candidateRows = page.locator('[data-testid="candidates-pane"] tbody tr')
      await expect(candidateRows.first()).toBeVisible({ timeout: 180000 })
      expect(await candidateRows.count()).toBeGreaterThan(0)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
      rmSync(cvDir, { recursive: true, force: true })
    }
  })
})
