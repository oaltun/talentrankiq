import { test, expect, type Page } from "@playwright/test"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Regression test for talentrankiq_gui/bug.md:
//
//  "On a fresh install (no scout/quick embedding model downloaded), the Job
//   Details page is unreachable: opening any job immediately redirects to the
//   Models page with reason=quick."
//
// Expected: the workspace (job ad, CV watch directory, candidates) renders even
// when the quick model is missing, exactly as it already does for a missing deep
// model. Ranking/import may be deferred, but the page must be reachable.
//
// The CLI is NOT mocked: the test points the dev proxy at an isolated real
// `triq serve` whose model dir is empty (fresh install), then drives the real UI.

const APP_DIR = () => mkdtempSync(join(tmpdir(), "triq-quick-bug-app-"))
// Intentionally empty model dir: no quick/scout model is downloaded.
const EMPTY_MODEL_DIR = () => mkdtempSync(join(tmpdir(), "triq-quick-bug-models-"))

async function useIsolatedCli(page: Page, appDir: string, modelDir: string) {
  const res = await page.request.post("/api/test-config", {
    data: { appDir, modelDir },
  })
  expect(res.ok()).toBeTruthy()
}

test.describe("bug.md — quick model redirect", () => {
  test("opening a job on a fresh install renders the workspace instead of redirecting to Models (reason=quick)", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = EMPTY_MODEL_DIR()

    try {
      // Fresh install: empty model cache, no static-similarity-mrl-multilingual-v1.
      await useIsolatedCli(page, appDir, modelDir)

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

      // The workspace must render even without the quick model downloaded.
      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 20000,
      })

      // Job setup panes must be usable.
      await expect(
        page.locator('[data-testid="workspace-left-pane"]')
      ).toBeVisible()
      await expect(
        page.locator('[data-testid="workspace-right-pane"]')
      ).toBeVisible()
      await expect(page.locator('[data-testid="cv-dir-input"]')).toBeVisible()

      // It must NOT have redirected to the Models page.
      await expect(page).not.toHaveURL(/#\/models\?.*reason=quick/)
      await expect(page.locator('[data-testid="models-content-quick"]')).toHaveCount(
        0
      )
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })
})