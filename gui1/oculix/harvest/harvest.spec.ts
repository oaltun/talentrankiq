import { test } from "@playwright/test"
import fs from "fs"
import path from "path"
import { getScenes, type Scene } from "./scenes"
import { computeRoi, buffersEqual, type BoundingBox } from "./roi"

const THEME = process.env.OCULIX_THEME
const SCENES_FILTER = process.env.OCULIX_SCENES ?? "all"

if (!THEME || (THEME !== "light" && THEME !== "dark")) {
  throw new Error("OCULIX_THEME must be set to 'light' or 'dark'")
}

const COUNTER_FILE = path.resolve("oculix/harvest/.run-counter")
const BASE_OUTPUT_DIR = path.resolve("oculix/img")

function readCounter(): number {
  try {
    const content = fs.readFileSync(COUNTER_FILE, "utf-8").trim()
    return parseInt(content, 10) || 0
  } catch {
    return 0
  }
}

function writeCounter(n: number): void {
  fs.mkdirSync(path.dirname(COUNTER_FILE), { recursive: true })
  fs.writeFileSync(COUNTER_FILE, String(n), "utf-8")
}

const runNumber = readCounter() + 1
writeCounter(runNumber)

const OUTPUT_DIR = path.join(BASE_OUTPUT_DIR, `img-${runNumber}`, THEME)
fs.mkdirSync(OUTPUT_DIR, { recursive: true })

const manifestPath = path.join(
  BASE_OUTPUT_DIR,
  `img-${runNumber}`,
  "manifest.json"
)
interface ManifestEntry {
  scene: string
  testid: string
  status: string
  path: string
}
const manifest: ManifestEntry[] = []

function saveManifest(): void {
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf-8")
}

function matchPrefix(testid: string, prefixes: string[]): boolean {
  return prefixes.some((prefix) => {
    if (prefix.endsWith("*")) {
      return testid.startsWith(prefix.slice(0, -1))
    }
    return testid === prefix
  })
}

async function captureIdle(page: Page, testid: string): Promise<Buffer> {
  const locator = page.locator(`[data-testid="${testid}"]`).first()
  await locator.scrollIntoViewIfNeeded()
  await page.waitForTimeout(100)
  return await page.screenshot({ fullPage: true, type: "png" })
}

async function captureAction(
  page: Page,
  testid: string,
  action: "hover" | "click" | "select"
): Promise<Buffer> {
  const locator = page.locator(`[data-testid="${testid}"]`).first()
  await locator.scrollIntoViewIfNeeded()

  if (action === "hover") {
    await locator.hover()
  } else if (action === "click") {
    await locator.click()
  } else if (action === "select") {
    await locator.focus()
    await page.keyboard.press("Control+a")
  }

  await page.waitForTimeout(100)
  return await page.screenshot({ fullPage: true, type: "png" })
}

async function getElementBBox(
  page: Page,
  testid: string
): Promise<BoundingBox | null> {
  const locator = page.locator(`[data-testid="${testid}"]`).first()
  const box = await locator.boundingBox()
  if (!box) return null
  return {
    x: Math.round(box.x),
    y: Math.round(box.y),
    width: Math.round(box.width),
    height: Math.round(box.height),
  }
}

async function harvestTestId(
  page: Page,
  scene: Scene,
  testid: string
): Promise<void> {
  const locator = page.locator(`[data-testid="${testid}"]`).first()
  if (!(await locator.isVisible().catch(() => false))) {
    console.warn(`  [${scene.name}] ${testid}: not visible, skipping`)
    return
  }

  const elementBBox = await getElementBBox(page, testid)
  if (!elementBBox) {
    console.warn(`  [${scene.name}] ${testid}: no bounding box, skipping`)
    return
  }

  const idleBuf = await captureIdle(page, testid)

  const capturedStatuses = new Set<string>()
  const actions: Array<"hover" | "click" | "select"> = [
    "hover",
    "click",
    "select",
  ]

  for (const action of actions) {
    await scene.setup(page)
    await page.waitForTimeout(200)

    const locator = page.locator(`[data-testid="${testid}"]`).first()
    if (!(await locator.isVisible().catch(() => false))) {
      continue
    }

    const activeBuf = await captureAction(page, testid, action)

    if (buffersEqual(idleBuf, activeBuf)) {
      continue
    }

    let alreadyCaptured = false
    for (const capturedStatus of capturedStatuses) {
      const capturedPath = path.join(
        OUTPUT_DIR,
        `${testid}-${capturedStatus}.png`
      )
      const capturedBuf = fs.readFileSync(capturedPath)
      if (buffersEqual(activeBuf, capturedBuf)) {
        alreadyCaptured = true
        break
      }
    }
    if (alreadyCaptured) {
      continue
    }

    const roi = await computeRoi(page, idleBuf, activeBuf, elementBBox)
    const cropped = await page.screenshot({ clip: roi, type: "png" })

    const filename = `${testid}-${action}.png`
    const filepath = path.join(OUTPUT_DIR, filename)
    fs.writeFileSync(filepath, cropped)

    manifest.push({
      scene: scene.name,
      testid,
      status: action,
      path: path.relative(path.dirname(manifestPath), filepath),
    })

    capturedStatuses.add(action)
    console.log(
      `  [${scene.name}] ${testid}: captured ${action} (${roi.width}x${roi.height})`
    )
  }

  if (capturedStatuses.size === 0) {
    const roi = await computeRoi(page, idleBuf, idleBuf, elementBBox)
    const cropped = await page.screenshot({ clip: roi, type: "png" })

    const filename = `${testid}-idle.png`
    const filepath = path.join(OUTPUT_DIR, filename)
    fs.writeFileSync(filepath, cropped)

    manifest.push({
      scene: scene.name,
      testid,
      status: "idle",
      path: path.relative(path.dirname(manifestPath), filepath),
    })

    console.log(
      `  [${scene.name}] ${testid}: captured idle (${roi.width}x${roi.height})`
    )
  }
}

test.describe("Oculix Harvest", () => {
  const selectedScenes = getScenes(SCENES_FILTER)

  for (const scene of selectedScenes) {
    test.describe(`Scene: ${scene.name}`, () => {
      test.beforeEach(async ({ page }) => {
        await scene.setup(page)
      })

      test(`harvest all testids for ${scene.name}`, async ({ page }) => {
        test.setTimeout(300000)
        const allTestIds = await page
          .locator("[data-testid]")
          .evaluateAll((els) =>
            Array.from(
              new Set(
                els
                  .map((el) => el.getAttribute("data-testid"))
                  .filter(Boolean) as string[]
              )
            ).sort()
          )

        const filteredTestIds = allTestIds.filter((tid) =>
          matchPrefix(tid, scene.testidPrefixes)
        )

        console.log(
          `[${scene.name}] Found ${filteredTestIds.length} testids to harvest`
        )

        for (const testid of filteredTestIds) {
          try {
            await harvestTestId(page, scene, testid)
          } catch (e) {
            console.error(`  [${scene.name}] ${testid}: ERROR - ${e}`)
          }
        }

        saveManifest()
      })
    })
  }
})
