import { expect, test } from "@playwright/test"

test("language selector in settings dialog changes language and persists", async ({
  page,
}) => {
  const pageErrors: string[] = []
  page.on("pageerror", (error) => pageErrors.push(error.message))

  await page.goto("/")

  // Open settings dialog
  await page.getByTestId("sidebar-secondary-settings-btn").click()
  await expect(page.getByTestId("settings-dialog")).toBeVisible()

  // Get the language select
  const languageSelect = page.getByTestId("language-select-trigger")
  await expect(languageSelect).toBeVisible()

  // Verify initial value is English
  await expect(languageSelect).toHaveValue("en")

  // Change to Turkish using selectOption
  await languageSelect.selectOption("tr")

  // Verify the language has changed to Turkish (Türkçe)
  // The select shows the value, but the display might show the option text
  await expect(languageSelect).toHaveValue("tr")

  // Close settings dialog using the close button
  await page.getByRole("button", { name: "Close" }).click()
  await expect(page.getByTestId("settings-dialog")).toBeHidden()

  // Reopen settings
  await page.getByTestId("sidebar-secondary-settings-btn").click()
  await expect(page.getByTestId("settings-dialog")).toBeVisible()

  // Verify language is still Turkish
  await expect(languageSelect).toHaveValue("tr")

  // Change back to English
  await languageSelect.selectOption("en")

  // Verify the language has changed to English
  await expect(languageSelect).toHaveValue("en")

  expect(pageErrors).toEqual([])
})
