import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'
import { FolderOpen } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { ShortcutPicker } from '../ShortcutPicker'
import { SettingsField, SettingsSection } from '../shared/SettingsComponents'
import { usePreferences, useSavePreferences } from '@/services/preferences'
import { commands } from '@/lib/tauri-bindings'
import { openFileDialog } from '@/lib/file-dialogs'
import { logger } from '@/lib/logger'

export function GeneralPane() {
  const { t } = useTranslation()
  // Example local state - these are NOT persisted to disk
  // To add persistent preferences:
  // 1. Add the field to AppPreferences in both Rust and TypeScript
  // 2. Use usePreferencesManager() and updatePreferences()
  const [exampleText, setExampleText] = useState('Example value')
  const [exampleToggle, setExampleToggle] = useState(true)

  // Load preferences for keyboard shortcuts
  const { data: preferences } = usePreferences()
  const savePreferences = useSavePreferences()

  // Effective CV-directory root while no custom root is saved (serve-side
  // resolution: gui preference > ikapp.yaml > platform Documents/jobs).
  const { data: defaultCvDirRoot } = useQuery({
    queryKey: ['cv-dir-root'],
    queryFn: async () => {
      const result = await commands.getCvDirRoot()
      return result.status === 'ok' ? result.data : null
    },
    staleTime: Infinity, // Constant per serve session; changes only via save below
  })

  // Get the default shortcut from the backend
  const { data: defaultShortcut } = useQuery({
    queryKey: ['default-quick-pane-shortcut'],
    queryFn: async () => {
      return await commands.getDefaultQuickPaneShortcut()
    },
    staleTime: Infinity, // Never refetch - this is a constant
  })

  const handleShortcutChange = async (newShortcut: string | null) => {
    if (!preferences) return

    // Capture old shortcut for rollback if save fails
    const oldShortcut = preferences.quick_pane_shortcut

    logger.info('Updating quick pane shortcut', { oldShortcut, newShortcut })

    // First, try to register the new shortcut
    const result = await commands.updateQuickPaneShortcut(newShortcut)

    if (result.status === 'error') {
      logger.error('Failed to register shortcut', { error: result.error })
      toast.error(t('toast.error.shortcutFailed'), {
        description: result.error,
      })
      return
    }

    // If registration succeeded, try to save the preference
    try {
      await savePreferences.mutateAsync({
        ...preferences,
        quick_pane_shortcut: newShortcut,
      })
    } catch {
      // Save failed - roll back the backend registration
      logger.warn('Save failed, rolling back shortcut registration', {
        oldShortcut,
        newShortcut,
      })

      const rollbackResult = await commands.updateQuickPaneShortcut(oldShortcut)

      if (rollbackResult.status === 'error') {
        logger.error(
          'Rollback failed - backend and preferences are out of sync',
          {
            error: rollbackResult.error,
            attemptedShortcut: newShortcut,
            originalShortcut: oldShortcut,
          }
        )
        toast.error(t('toast.error.shortcutRestoreFailed'), {
          description: t('toast.error.shortcutRestoreDescription'),
        })
      } else {
        logger.info('Successfully rolled back shortcut registration')
      }
    }
  }

  const shownCvDirRoot = preferences?.cv_dir_root ?? defaultCvDirRoot ?? ''
  const isCustomCvDirRoot = !!preferences?.cv_dir_root

  const handleCvDirRootChange = async () => {
    if (!preferences) return
    const dir = await openFileDialog({
      directory: true,
      multiple: false,
      startDir: shownCvDirRoot || undefined,
    })
    if (typeof dir !== 'string' || dir === shownCvDirRoot) return
    savePreferences.mutate({ ...preferences, cv_dir_root: dir })
  }

  const handleCvDirRootReset = () => {
    if (!preferences) return
    savePreferences.mutate({ ...preferences, cv_dir_root: null })
  }

  return (
    <div className="space-y-6">
      <SettingsSection title={t('preferences.general.cvDirectory')}>
        <SettingsField
          label={t('preferences.general.cvDirectoryRoot')}
          description={t('preferences.general.cvDirectoryRootDescription')}
        >
          <div className="flex w-full flex-col gap-2">
            <div
              className="min-w-0 rounded-md border px-3 py-1 font-mono text-xs break-all text-muted-foreground select-text"
              title={
                isCustomCvDirRoot
                  ? undefined
                  : t('preferences.general.cvDirectoryDefaultHint')
              }
            >
              {shownCvDirRoot || (
                <span className="italic">
                  {t('preferences.general.cvDirectoryUnavailable')}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCvDirRootChange}
                disabled={!preferences || savePreferences.isPending}
              >
                <FolderOpen className="size-4" />
                {t('preferences.general.cvDirectoryChange')}
              </Button>
              {isCustomCvDirRoot && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCvDirRootReset}
                  disabled={savePreferences.isPending}
                >
                  {t('preferences.general.cvDirectoryReset')}
                </Button>
              )}
              <span className="text-xs text-muted-foreground">
                {isCustomCvDirRoot
                  ? t('preferences.general.cvDirectoryCustom')
                  : t('preferences.general.cvDirectoryDefault')}
              </span>
            </div>
          </div>
        </SettingsField>
      </SettingsSection>

      <SettingsSection title={t('preferences.general.keyboardShortcuts')}>
        <SettingsField
          label={t('preferences.general.quickPaneShortcut')}
          description={t('preferences.general.quickPaneShortcutDescription')}
        >
          <ShortcutPicker
            value={preferences?.quick_pane_shortcut ?? null}
            // Fallback matches DEFAULT_QUICK_PANE_SHORTCUT in src-tauri/src/lib.rs
            defaultValue={defaultShortcut ?? 'CommandOrControl+Shift+.'}
            onChange={handleShortcutChange}
            disabled={!preferences || savePreferences.isPending}
          />
        </SettingsField>
      </SettingsSection>

      <SettingsSection title={t('preferences.general.exampleSettings')}>
        <SettingsField
          label={t('preferences.general.exampleText')}
          description={t('preferences.general.exampleTextDescription')}
        >
          <Input
            value={exampleText}
            onChange={e => setExampleText(e.target.value)}
            placeholder={t('preferences.general.exampleTextPlaceholder')}
          />
        </SettingsField>

        <SettingsField
          label={t('preferences.general.exampleToggle')}
          description={t('preferences.general.exampleToggleDescription')}
        >
          <div className="flex items-center space-x-2">
            <Switch
              id="example-toggle"
              checked={exampleToggle}
              onCheckedChange={setExampleToggle}
            />
            <Label htmlFor="example-toggle" className="text-sm">
              {exampleToggle ? t('common.enabled') : t('common.disabled')}
            </Label>
          </div>
        </SettingsField>
      </SettingsSection>
    </div>
  )
}
