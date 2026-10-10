import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { GuiEmbedder } from '@/lib/tauri-bindings'
import { useEmbedders, useSetEmbeddingModel } from '@/services/models'
import { SettingsField, SettingsSection } from '../shared/SettingsComponents'

function fmtSize(bytes: number): string {
  if (!bytes) return ''
  const gb = bytes / 1024 ** 3
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`
}

/**
 * Global embedding-model selector. The model decides how CVs, job ads,
 * criteria and learned feedback are vectorized — app-wide, not per job:
 * each model's vectors and feedback live separately, and switching empties
 * every job's candidates (re-import needed) and re-embeds all job ads and
 * criteria with the new model. Hence its home in Preferences, not in any
 * job's workspace.
 */
export function ModelsPane() {
  const { t } = useTranslation()
  const { data: embedders = [] } = useEmbedders()
  const setModel = useSetEmbeddingModel()
  const [pending, setPending] = useState<GuiEmbedder | null>(null)
  const active = embedders.find(e => e.is_default)

  return (
    <>
      <SettingsSection title={t('preferences.models.embedding')}>
        <SettingsField
          label={t('preferences.models.embeddingModel')}
          description={t('preferences.models.embeddingModelDescription')}
        >
          <Select
            value={active?.slug ?? ''}
            onValueChange={slug =>
              setPending(embedders.find(e => e.slug === slug) ?? null)
            }
            disabled={setModel.isPending}
          >
            <SelectTrigger className="w-full">
              <SelectValue
                placeholder={t('preferences.models.choosePlaceholder')}
              />
            </SelectTrigger>
            <SelectContent>
              {embedders.map(e => (
                <SelectItem
                  key={e.slug}
                  value={e.slug}
                  disabled={!e.downloaded}
                >
                  <span className="flex flex-col">
                    <span>{e.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {t('preferences.models.dim', { dim: e.dim })}
                      {e.downloaded && e.size_bytes
                        ? ` · ${fmtSize(e.size_bytes)}`
                        : ''}
                      {!e.downloaded &&
                        ` · ${t('preferences.models.notDownloaded')}`}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsField>
      </SettingsSection>

      <AlertDialog
        open={!!pending}
        onOpenChange={o => {
          if (!o) setPending(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t('preferences.models.switchTitle', {
                name: pending?.name ?? '',
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t('preferences.models.switchDescription')}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t('preferences.models.switchCancel')}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pending) setModel.mutate(pending.slug)
                setPending(null)
              }}
            >
              {t('preferences.models.switchAction')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
