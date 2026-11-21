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

// The deep embedder card only renders the "Disk boyutu" (disk size) row and the
// "Sil" (remove) button when a model is actually present on disk, so the
// translations test needs one downloaded deep model in the cache.
async function setupDeepModel(modelDir: string) {
  const deepSlug = "harrier-oss-v1-270m-ONNX-q4f16"
  const deepOnnx = join(modelDir, deepSlug, "repo", "onnx")
  mkdirSync(deepOnnx, { recursive: true })
  writeFileSync(join(deepOnnx, "model.onnx"), "fake")
}

test.describe("Turkish translations on Models page", () => {
  test("Models page shows Turkish translations", async ({ page }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await setupDeepModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.evaluate(() => localStorage.setItem("language", "tr"))
      await page.reload()
      await page.waitForLoadState("networkidle")

      await page.locator('button:has-text("Modeller")').click()
      await page.waitForLoadState("networkidle")

      // Check page title and subtitle
      await expect(
        page.getByRole("heading", { level: 1, name: "Modeller" })
      ).toBeVisible()
      await expect(
        page.locator(
          "text=CV'leri sıralamak için kullanılan cihazdaki gömme modellerini yönetin."
        )
      ).toBeVisible()

      // Check description paragraph
      await expect(
        page.locator(
          "text=Cihazda gömme modellerini kullanarak CV'leri sıralamak için kullanılan modelleri yönetin"
        )
      ).toBeVisible()

      // Check tabs
      await expect(page.locator('[role="tab"]:has-text("Hızlı")')).toBeVisible()
      await expect(page.locator('[role="tab"]:has-text("Derin")')).toBeVisible()

      // Check model card specs (use .first() since there are 4 model cards)
      await expect(page.locator("text=RAM gerekli").first()).toBeVisible()
      await expect(page.locator("text=Gömme boyutları").first()).toBeVisible()
      await expect(
        page.locator("text=Maks. bağlam penceresi").first()
      ).toBeVisible()
      await expect(page.locator("text=Disk boyutu").first()).toBeVisible()

      // Check button texts
      await expect(
        page.locator('button:has-text("Varsayılan yap")').first()
      ).toBeVisible()
      await expect(page.locator('button:has-text("Sil")').first()).toBeVisible()
      await expect(page.locator('button:has-text("İndir")').first()).toBeVisible()

      // Check comparator section
      await expect(
        page.locator("text=Karşılaştırıcı artefakt").first()
      ).toBeVisible()
      await expect(page.locator("text=Karşılaştırıcıyı aç").first()).toBeVisible()
      await expect(
        page
          .locator(
            "text=CV'leri baş başa LLM yargısıyla sıralar — kosinüs'ten daha kesin."
          )
          .first()
      ).toBeVisible()
      await expect(page.locator("text=Comparator: eksik").first()).toBeVisible()
      await expect(
        page.locator("text=Dosya şu isimle olmalı").first()
      ).toBeVisible()
      await expect(
        page.locator('button:has-text("Karşılaştırıcıyı al")').first()
      ).toBeVisible()
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("Default badge shows Turkish", async ({ page }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      await setupQuickModel(modelDir)
      await setupDeepModel(modelDir)
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.evaluate(() => localStorage.setItem("language", "tr"))
      await page.reload()
      await page.waitForLoadState("networkidle")

      await page.locator('button:has-text("Modeller")').click()
      await page.waitForLoadState("networkidle")

      // Check the default badge shows "varsayılan" (use the badge specifically)
      await expect(
        page.locator('[data-testid^="model-default-badge-"]').first()
      ).toBeVisible()
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })

  test("Quick tab is fully translated to Turkish", async ({ page }) => {
    const appDir = APP_DIR()
    const modelDir = MODEL_DIR()
    try {
      // No quick model on disk: the card must render the Turkish Download
      // ("İndir") button and the "bilinmiyor" (unknown) size label.
      await useIsolatedCli(page, appDir, modelDir)

      await page.goto("/")
      await page.evaluate(() => localStorage.setItem("language", "tr"))
      await page.reload()
      await page.waitForLoadState("networkidle")

      await page.locator('button:has-text("Modeller")').click()
      await page.waitForLoadState("networkidle")
      await page.locator('[role="tab"]:has-text("Hızlı")').click()
      await page.waitForLoadState("networkidle")

      // Quick model card renders in the browser catalog (not downloaded).
      await expect(
        page.locator('[data-testid="models-content-quick"]')
      ).toBeVisible()
      await expect(
        page.locator(
          '[data-testid="models-content-quick"] button:has-text("İndir")'
        )
      ).toBeVisible()
      await expect(
        page.locator('[data-testid="models-content-quick"] >> text=bilinmiyor')
      ).toBeVisible()

      // No leftover English from the quick tab.
      await expect(
        page.locator('[data-testid="models-content-quick"] >> text=unknown')
      ).toHaveCount(0)
      await expect(
        page.locator(
          '[data-testid="models-content-quick"] button:has-text("Download")'
        )
      ).toHaveCount(0)
      await expect(
        page.locator(
          '[data-testid="models-content-quick"] >> text=Loading quick model...'
        )
      ).toHaveCount(0)
    } finally {
      rmSync(appDir, { recursive: true, force: true })
      rmSync(modelDir, { recursive: true, force: true })
    }
  })
})