import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { SimpleSelect } from "@/components/settings/simple-select"
import { LANGUAGES, LANGUAGE_NAMES, useI18n, type Language } from "@/i18n"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { t, language, setLanguage } = useI18n()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-testid="settings-dialog">
        <DialogHeader>
          <DialogTitle data-testid="settings-dialog-title">
            {t("settings.title")}
          </DialogTitle>
          <DialogDescription>
            {t("settings.languageDescription")}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">{t("settings.language")}</span>
          <SimpleSelect
            value={language}
            onValueChange={(value) => {
              if (value !== null) {
                setLanguage(value as Language)
              }
            }}
            data-testid="language-select-trigger"
            aria-label={t("settings.language")}
            className="relative w-full"
          >
            {LANGUAGES.map((code) => (
              <SimpleSelect.Option
                key={code}
                value={code}
                data-testid={`language-option-${code}`}
              >
                {LANGUAGE_NAMES[code]}
              </SimpleSelect.Option>
            ))}
          </SimpleSelect>
        </div>
      </DialogContent>
    </Dialog>
  )
}
