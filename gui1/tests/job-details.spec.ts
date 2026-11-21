import { test, expect } from "@playwright/test"
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

const APP_DIR = () => mkdtempSync(join(tmpdir(), "triq-e2e-app-"))
const MODEL_DIR = () => mkdtempSync(join(tmpdir(), "triq-e2e-models-"))

async function useIsolatedCli(
  page: import("@playwright/test").Page,
  appDir: string,
  modelDir: string
) {
  const res = await page.request.post("/api/test-config", {
    data: { appDir, modelDir },
  })
  expect(res.ok()).toBeTruthy()
}

async function setupQuickModel(modelDir: string) {
  const quickSlug = "static-similarity-mrl-multilingual-v1"
  const quickOnnx = join(modelDir, quickSlug, "repo", "onnx")
  mkdirSync(quickOnnx, { recursive: true })
  writeFileSync(join(quickOnnx, "model.onnx"), "fake")
}

test.describe("job details", () => {
  test("creating a job and opening it does not render a blank page", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()

      await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
        timeout: 10000,
      })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 15000,
      })

      await expect(
        page.locator('[data-testid="workspace-left-pane"]')
      ).toBeVisible()
      await expect(
        page.locator('[data-testid="workspace-right-pane"]')
      ).toBeVisible()

      const pageErrors: string[] = []
      page.on("pageerror", (error) => pageErrors.push(error.message))
      expect(pageErrors).toEqual([])
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("cv watch directory change button renders above the input in the header", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()

      await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
        timeout: 10000,
      })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 15000,
      })

      const changeBtn = page.locator('[data-testid="change-cv-dir-btn"]')
      const input = page.locator('[data-testid="cv-dir-input"]')
      await expect(changeBtn).toBeVisible()
      await expect(input).toBeVisible()

      const btnBox = await changeBtn.boundingBox()
      const inputBox = await input.boundingBox()
      expect(btnBox).not.toBeNull()
      expect(inputBox).not.toBeNull()
      expect(btnBox!.y + btnBox!.height).toBeLessThanOrEqual(inputBox!.y + 1)

      const pageErrors: string[] = []
      page.on("pageerror", (error) => pageErrors.push(error.message))
      expect(pageErrors).toEqual([])
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("job ad card renders the path as an input and drops content + reload UI", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()

      await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
        timeout: 10000,
      })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 15000,
      })

      await expect(page.locator('[data-testid="ad-path-input"]')).toBeVisible({
        timeout: 15000,
      })

      await expect(page.locator('[data-testid="reload-ad-btn"]')).toHaveCount(0)

      const pageErrors: string[] = []
      page.on("pageerror", (error) => pageErrors.push(error.message))
      expect(pageErrors).toEqual([])
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("cv watch directory input shows absolute path with leading slash", async ({
    page,
  }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.waitForLoadState("networkidle")

      await page.locator('[data-testid="add-job-button"]').click()

      await expect(page.locator('[data-testid^="job-item-"]').first()).toBeVisible({
        timeout: 10000,
      })

      const jobItems = page
        .locator('[data-testid="job-list"]')
        .locator('[data-testid^="job-item-"]')
      await jobItems.first().click()

      await expect(page.locator('[data-testid="workspace-page"]')).toBeVisible({
        timeout: 15000,
      })

      await expect(page.locator('[data-testid="cv-dir-input"]')).toBeVisible()

      const cvDirValue = await page
        .locator('[data-testid="cv-dir-input"]')
        .inputValue()

      expect(cvDirValue.startsWith("/")).toBeTruthy()

      const pageErrors: string[] = []
      page.on("pageerror", (error) => pageErrors.push(error.message))
      expect(pageErrors).toEqual([])
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })
})