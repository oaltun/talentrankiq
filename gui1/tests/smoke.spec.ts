import { expect, test } from "@playwright/test"

test("app shell renders without crashing", async ({ page }) => {
  const pageErrors: string[] = []
  page.on("pageerror", (error) => pageErrors.push(error.message))

  await page.goto("/")

  await expect(
    page.getByRole("heading", { level: 1, name: "Jobs" })
  ).toBeVisible()
  await expect(page.locator("[data-ui-scroll-container]")).toBeVisible()

  await page.keyboard.press("d")
  await expect(page.locator("html")).toHaveClass(/\bdark\b/)

  expect(pageErrors).toEqual([])
})

test("sidebar navigates to models page", async ({ page }) => {
  const pageErrors: string[] = []
  page.on("pageerror", (error) => pageErrors.push(error.message))

  await page.goto("/#/")
  await page.getByRole("button", { name: "Models" }).click()
  await expect(
    page.getByRole("heading", { level: 1, name: "Models" })
  ).toBeVisible()

  expect(pageErrors).toEqual([])
})

test("page text is selectable and copyable", async ({ page }) => {
  await page.goto("/")

  const bodyUserSelect = await page.evaluate(
    () => getComputedStyle(document.body).userSelect
  )
  expect(bodyUserSelect).not.toBe("none")

  const selection = await page.evaluate(() => {
    const heading = document.querySelector("h1")
    if (!heading || !heading.firstChild) return ""
    const range = document.createRange()
    range.selectNodeContents(heading)
    const sel = window.getSelection()
    sel?.removeAllRanges()
    sel?.addRange(range)
    return sel?.toString() ?? ""
  })
  expect(selection.trim().length).toBeGreaterThan(0)
})
