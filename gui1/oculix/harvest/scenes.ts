import type { Page } from "@playwright/test"

export type Scene = {
  name: string
  testidPrefixes: string[]
  setup: (page: Page) => Promise<void>
}

async function waitForNetworkIdle(page: Page) {
  await page.waitForLoadState("networkidle")
}

async function clearDatabase(page: Page) {
  await page.request.post("/api/invoke", {
    data: { cmd: "clear_all_data" },
    headers: { "Content-Type": "application/json" },
  })
}

async function goHome(page: Page) {
  await page.locator('[data-testid="activity-home"]').click()
  await waitForNetworkIdle(page)
}

async function goModels(page: Page) {
  await page.locator('[data-testid="activity-models"]').click()
  await waitForNetworkIdle(page)
}

async function clickAddJobButton(page: Page) {
  const addJobBtn = page.locator('[data-testid="add-job-button"]')
  await addJobBtn.click()
  await expectJobItemVisible(page)
}

async function expectJobItemVisible(page: Page) {
  await page
    .locator('[data-testid^="job-item-"]')
    .first()
    .waitFor({ state: "visible", timeout: 10000 })
}

async function clickFirstJobItem(page: Page) {
  await page.locator('[data-testid^="job-item-"]').first().click()
  await waitForNetworkIdle(page)
}

async function downloadQuickModel(page: Page) {
  await goModels(page)
  await page.locator('[data-testid="models-tab-quick"]').click()
  await expect(
    page.locator('[data-testid="models-tab-quick"]')
  ).toHaveAttribute("aria-selected", "true")

  const downloadButtons = page.locator('button[data-testid^="model-download-"]')
  await downloadButtons.first().click()

  await page
    .locator('button[data-testid^="model-remove-"]')
    .first()
    .waitFor({ state: "visible", timeout: 180000 })
}

async function downloadDeepModel(page: Page) {
  await goModels(page)
  await page.locator('[data-testid="models-tab-deep"]').click()
  await expect(page.locator('[data-testid="models-tab-deep"]')).toHaveAttribute(
    "aria-selected",
    "true"
  )

  const downloadButtons = page.locator('button[data-testid^="model-download-"]')
  await downloadButtons.first().click()

  await page
    .locator('button[data-testid^="model-remove-"]')
    .first()
    .waitFor({ state: "visible", timeout: 180000 })
}

async function setupCvWatchDirectory(page: Page) {
  const testCvDir =
    "/Users/apple1/Documents/repo/act/ik/activeik/talentrankiq_private/test-cv/Job10CVs/cv"
  const cvDirInput = page.locator('[data-testid="cv-dir-input"]')
  await cvDirInput.click()
  await cvDirInput.fill(testCvDir)
  await expect(cvDirInput).toHaveValue(testCvDir)

  await page.locator('[data-testid="watch-cv-dir-switch"]').click()
  await waitForNetworkIdle(page)
}

async function waitForConvertEnabled(page: Page) {
  await page
    .locator('[data-testid="convert-btn"]')
    .waitFor({ state: "enabled", timeout: 120000 })
}

async function openCvInspectorDialog(page: Page) {
  await page
    .locator('[data-testid="candidates-pane"] >> table >> tbody >> tr')
    .first()
    .click()
  await page
    .locator('[role="dialog"]')
    .waitFor({ state: "visible", timeout: 10000 })
}

const homeEmptyScene: Scene = {
  name: "home-empty",
  testidPrefixes: [
    "add-job-button-empty",
    "add-job-button",
    "activity-home",
    "app-sidebar-*",
    "infobar*",
    "sidebar-*",
  ],
  setup: async (page) => {
    await clearDatabase(page)
    await page.goto("/")
    await waitForNetworkIdle(page)
  },
}

const homeWithJobScene: Scene = {
  name: "home-with-job",
  testidPrefixes: ["job-item-*", "job-list"],
  setup: async (page) => {
    await clearDatabase(page)
    await page.goto("/")
    await waitForNetworkIdle(page)
    await clickAddJobButton(page)
  },
}

const modelsScene: Scene = {
  name: "models",
  testidPrefixes: [
    "models-tab-*",
    "models-content-*",
    "model-remove-*",
    "model-download-*",
    "specs-grid-*",
    "card-actions-*",
    "comparator-*",
  ],
  setup: async (page) => {
    await clearDatabase(page)
    await page.goto("/")
    await waitForNetworkIdle(page)
    await goModels(page)
  },
}

const workspaceReadyScene: Scene = {
  name: "workspace-ready",
  testidPrefixes: [
    "workspace-*",
    "ad-file-card",
    "job-ad-label",
    "ad-file-path",
    "change-ad-btn",
    "reload-ad-btn",
    "cv-directory-label",
    "cv-dir-input",
    "watch-cv-dir-switch",
    "change-cv-dir-btn",
    "open-cv-dir-btn",
    "conversion-progress",
    "convert-btn",
    "quick-progress",
    "quick-embed-btn",
    "deep-progress",
    "deep-embed-btn",
    "candidates-pane",
    "candidates-label",
  ],
  setup: async (page) => {
    await clearDatabase(page)
    await page.goto("/")
    await waitForNetworkIdle(page)

    await clickAddJobButton(page)
    await clickFirstJobItem(page)

    await page.locator('[data-testid="infobar"]').waitFor({ state: "visible" })
    await page.locator('[data-testid="infobar-action"]').click()

    await page
      .locator('[data-testid="models-tab-quick"]')
      .waitFor({ state: "attached" })
    await expect(
      page.locator('[data-testid="models-tab-quick"]')
    ).toHaveAttribute("aria-selected", "true")

    await downloadQuickModel(page)

    await goHome(page)
    await clickFirstJobItem(page)

    await page.locator('[data-testid="infobar"]').waitFor({ state: "visible" })
    await page.locator('[data-testid="infobar-action"]').click()

    await page
      .locator('[data-testid="models-tab-deep"]')
      .waitFor({ state: "attached" })
    await expect(
      page.locator('[data-testid="models-tab-deep"]')
    ).toHaveAttribute("aria-selected", "true")

    await downloadDeepModel(page)

    await goHome(page)
    await clickFirstJobItem(page)

    await page
      .locator('[data-testid="workspace-page"]')
      .waitFor({ state: "visible" })

    await setupCvWatchDirectory(page)
    await waitForConvertEnabled(page)
  },
}

const dialogsScene: Scene = {
  name: "dialogs",
  testidPrefixes: [
    "cv-inspector-*",
    "rubric-*",
    "match-lens-*",
    "highlighted-cv-*",
  ],
  setup: async (page) => {
    await clearDatabase(page)
    await page.goto("/")
    await waitForNetworkIdle(page)

    await clickAddJobButton(page)
    await clickFirstJobItem(page)

    await page.locator('[data-testid="infobar"]').waitFor({ state: "visible" })
    await page.locator('[data-testid="infobar-action"]').click()
    await downloadQuickModel(page)

    await goHome(page)
    await clickFirstJobItem(page)
    await page.locator('[data-testid="infobar"]').waitFor({ state: "visible" })
    await page.locator('[data-testid="infobar-action"]').click()
    await downloadDeepModel(page)

    await goHome(page)
    await clickFirstJobItem(page)
    await page
      .locator('[data-testid="workspace-page"]')
      .waitFor({ state: "visible" })

    await setupCvWatchDirectory(page)
    await waitForConvertEnabled(page)

    await page.locator('[data-testid="convert-btn"]').click()
    await page
      .locator('[data-testid="convert-btn"]')
      .waitFor({ state: "disabled", timeout: 120000 })

    await page.locator('[data-testid="quick-embed-btn"]').click()
    await page.waitForTimeout(10000)

    await page.locator('[data-testid="deep-embed-btn"]').click()
    await page.waitForTimeout(10000)

    await openCvInspectorDialog(page)
  },
}

export const scenes: Scene[] = [
  homeEmptyScene,
  homeWithJobScene,
  modelsScene,
  workspaceReadyScene,
  dialogsScene,
]

export function getScenes(names?: string): Scene[] {
  if (!names || names === "all") {
    return scenes
  }
  const requested = names.split(",").map((s) => s.trim())
  return scenes.filter((s) => requested.includes(s.name))
}
