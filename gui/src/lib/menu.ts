/**
 * Application menu.
 *
 * - **macOS**: installed as the native menubar (`setAsAppMenu`), grouped into
 *   the standard App / File / View / Window menus, so the items live in
 *   the system menubar like every other macOS app and there is no in-app
 *   hamburger at all. There is deliberately **no Edit menu**: the app has no
 *   text editor of its own any more (the job ad is a file the user edits in
 *   their own editor), so Cut/Copy/Paste/Undo belonged to a textarea that is
 *   gone. Clipboard shortcuts still work where macOS routes them itself.
 * - **Linux / Windows**: kept as a flat popup that the activity bar's hamburger
 *   opens, because neither platform has a reliably visible global menubar.
 *
 * Built from JavaScript for i18n support and rebuilt when the language changes.
 */
import {
  Menu,
  MenuItem,
  PredefinedMenuItem,
  Submenu,
} from '@tauri-apps/api/menu'
import { LogicalPosition } from '@tauri-apps/api/dpi'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { toast } from 'sonner'
import i18n from '@/i18n/config'
import { useUIStore } from '@/store/ui-store'
import { logger } from '@/lib/logger'
import { openFileDialog } from '@/lib/file-dialogs'
import { commands, unwrapResult } from '@/lib/tauri-bindings'
import { errorMessage } from '@/lib/errors'
import { getPlatform } from '@/hooks/use-platform'
import { jobsQueryKeys } from '@/services/jobs'

const APP_NAME = 'TalentRankIQ'

/** The currently built menu (rebuilt on language change). */
let currentMenu: Menu | null = null

/** The flat popup menu used by the hamburger on Linux/Windows. */
async function buildPopupMenu(): Promise<Menu> {
  const t = i18n.t.bind(i18n)

  return Menu.new({
    items: [
      await MenuItem.new({
        id: 'about',
        text: t('menu.about', { appName: APP_NAME }),
        action: handleAbout,
      }),
      await PredefinedMenuItem.new({ item: 'Separator' }),
      await MenuItem.new({
        id: 'import-job',
        text: t('menu.importJob'),
        action: handleImportJob,
      }),
      await PredefinedMenuItem.new({ item: 'Separator' }),
      await MenuItem.new({
        id: 'preferences',
        text: t('menu.preferences'),
        accelerator: 'CmdOrCtrl+,',
        action: handleOpenPreferences,
      }),
      await PredefinedMenuItem.new({ item: 'Separator' }),
      await PredefinedMenuItem.new({
        item: 'Quit',
        text: t('menu.quit', { appName: APP_NAME }),
      }),
    ],
  })
}

/**
 * The native macOS menubar. A macOS app menu may only contain submenus, and
 * the first one is the application menu (About / Hide / Quit). Predefined items
 * are left untitled where macOS already ships a localized title, so the OS
 * language wins for Undo/Cut/Copy/Minimize/Close/Fullscreen.
 */
async function buildMacMenu(): Promise<Menu> {
  const t = i18n.t.bind(i18n)
  const separator = () => PredefinedMenuItem.new({ item: 'Separator' })

  const windowMenu = await Submenu.new({
    text: t('menu.window'),
    items: [
      await PredefinedMenuItem.new({ item: 'Minimize' }),
      await PredefinedMenuItem.new({ item: 'Maximize' }),
      await separator(),
      await PredefinedMenuItem.new({ item: 'BringAllToFront' }),
    ],
  })

  const menu = await Menu.new({
    items: [
      await Submenu.new({
        text: APP_NAME,
        items: [
          await MenuItem.new({
            id: 'about',
            text: t('menu.about', { appName: APP_NAME }),
            action: handleAbout,
          }),
          await separator(),
          await MenuItem.new({
            id: 'preferences',
            text: t('menu.preferences'),
            accelerator: 'CmdOrCtrl+,',
            action: handleOpenPreferences,
          }),
          await separator(),
          await PredefinedMenuItem.new({ item: 'Services' }),
          await separator(),
          await PredefinedMenuItem.new({
            item: 'Hide',
            text: t('menu.hide', { appName: APP_NAME }),
          }),
          await PredefinedMenuItem.new({
            item: 'HideOthers',
            text: t('menu.hideOthers'),
          }),
          await PredefinedMenuItem.new({
            item: 'ShowAll',
            text: t('menu.showAll'),
          }),
          await separator(),
          await PredefinedMenuItem.new({
            item: 'Quit',
            text: t('menu.quit', { appName: APP_NAME }),
          }),
        ],
      }),

      await Submenu.new({
        text: t('menu.file'),
        items: [
          await MenuItem.new({
            id: 'import-job',
            text: t('menu.importJob'),
            accelerator: 'CmdOrCtrl+O',
            action: handleImportJob,
          }),
          await separator(),
          await PredefinedMenuItem.new({ item: 'CloseWindow' }),
        ],
      }),

      await Submenu.new({
        text: t('menu.view'),
        items: [await PredefinedMenuItem.new({ item: 'Fullscreen' })],
      }),

      windowMenu,
    ],
  })

  // Let macOS manage window switching/minimize placement inside the Window menu.
  // Best-effort: losing it only costs the automatic window list, so never let it
  // discard an otherwise-good menubar.
  try {
    await windowMenu.setAsWindowsMenuForNSApp()
  } catch (error) {
    logger.warn('Could not register the Window menu with NSApp', { error })
  }

  return menu
}

/**
 * Build (and cache) the application menu. On macOS it is installed as the
 * native menubar; elsewhere it stays unassigned until the hamburger pops it.
 */
export async function buildAppMenu(): Promise<Menu | null> {
  try {
    const isMacOS = getPlatform() === 'macos'
    const menu = isMacOS ? await buildMacMenu() : await buildPopupMenu()

    if (isMacOS) {
      await menu.setAsAppMenu()
    }

    currentMenu = menu
    logger.info('Application menu built successfully')
    return menu
  } catch (error) {
    logger.error('Failed to build application menu', { error })
    throw error
  }
}

/** Pop up the application menu at the given viewport position (hamburger, Linux/Windows). */
export async function popupAppMenu(viewportPosition?: {
  x: number
  y: number
}): Promise<void> {
  if (getPlatform() === 'macos') {
    // The menu is the native menubar there; there is nothing to pop up.
    return
  }

  const show = async (): Promise<void> => {
    // Always popup a freshly-built menu: a Menu cached since startup can go
    // stale on the Rust side (GTK grabs, dropped resources on Linux), which
    // left the hamburger button dead until the app was restarted. Building is
    // a handful of cheap IPC calls, so always popup a fresh one.
    await buildAppMenu()
    if (!currentMenu) return

    if (viewportPosition) {
      // Convert viewport coordinates to window-relative logical coordinates
      const window = getCurrentWindow()
      const innerPosition = await window.innerPosition()
      const scaleFactor = await window.scaleFactor()

      // viewportPosition is relative to the viewport (browser client area)
      // innerPosition is the window's client area position on screen (physical pixels)
      // We need to convert viewport position to window-relative logical pixels
      const logicalX = (viewportPosition.x - innerPosition.x) / scaleFactor
      const logicalY = (viewportPosition.y - innerPosition.y) / scaleFactor

      await currentMenu.popup(new LogicalPosition(logicalX, logicalY))
    } else {
      await currentMenu.popup()
    }
  }

  try {
    await show()
  } catch (error) {
    // One retry from a clean build — recovers one-off failures (e.g. a GTK
    // grab from a just-closed native menu blocking the new popup).
    logger.warn('Menu popup failed, retrying from a clean build', { error })
    currentMenu = null
    try {
      await show()
    } catch (retryError) {
      logger.error('Failed to popup application menu', { error: retryError })
      currentMenu = null
    }
  }
}

/**
 * Set up a listener to rebuild the menu when the language changes.
 * Returns an unsubscribe function for cleanup.
 */
export function setupMenuLanguageListener(): () => void {
  const handler = async () => {
    logger.info('Language changed, rebuilding menu')
    try {
      await buildAppMenu()
    } catch (error) {
      logger.error('Failed to rebuild menu on language change', { error })
    }
  }
  i18n.on('languageChanged', handler)
  return () => i18n.off('languageChanged', handler)
}

// Menu action handlers

function handleAbout(): void {
  logger.info('About menu item clicked')
  alert(
    `${APP_NAME}\n\nVersion: ${__APP_VERSION__}\n\nBuilt with Tauri v2 + React + TypeScript`
  )
}

function handleOpenPreferences(): void {
  logger.info('Preferences menu item clicked')
  useUIStore.getState().setPreferencesOpen(true)
}

/**
 * Import a `.triquijob` bundle as a new job, then navigate to it. The
 * backend creates the job, stages the CV files and runs the import pipeline;
 * progress shows up in the job's setup pane like a manual CV import.
 */
async function handleImportJob(): Promise<void> {
  logger.info('Import job menu item clicked')
  try {
    const path = await openFileDialog({
      filters: [{ name: 'TalentRankIQ job', extensions: ['triquijob'] }],
    })
    if (typeof path !== 'string') return
    const job = unwrapResult(await commands.importJobBundle(path))
    // Dynamic import: menu.ts is loaded from the router's component tree, so a
    // static import would create a module cycle.
    const { router } = await import('@/router')
    const { queryClient } = await import('@/lib/query-client')
    queryClient.invalidateQueries({ queryKey: jobsQueryKeys.all })
    await router.navigate({ to: '/jobs/$jobId', params: { jobId: job.id } })
    toast.success(`Imported "${job.title}" — CVs are being imported`)
  } catch (error) {
    logger.error('Job import failed', { error })
    toast.error(`Import failed: ${errorMessage(error)}`)
  }
}
