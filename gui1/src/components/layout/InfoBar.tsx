import { Info, AlertCircle, X } from "lucide-react"
import { useI18n } from "@/i18n"
import { Button } from "../ui/button"

interface InfoBarProps {
  isVisible: boolean
  variant?: "info" | "warning" | "error"
  title: string
  description: string
  actionLabel?: string
  onAction?: () => void
  onDismiss?: () => void
}

export function InfoBar({
  isVisible,
  variant = "info",
  title,
  description,
  actionLabel,
  onAction,
  onDismiss,
}: InfoBarProps) {
  const { t } = useI18n()

  if (!isVisible) return null

  const variantStyles = {
    info: {
      container: "border-primary/40 bg-primary/10",
      icon: "text-primary",
    },
    warning: {
      container: "border-yellow-500/40 bg-yellow-500/10",
      icon: "text-yellow-500",
    },
    error: {
      container: "border-destructive bg-destructive/10",
      icon: "text-destructive",
    },
  }[variant]

  const Icon = variant === "info" ? Info : AlertCircle

  return (
    <div className="mx-auto max-w-3xl px-4">
      <div
        className={`flex items-start gap-3 rounded-lg border p-4 shadow-md backdrop-blur-sm ${variantStyles.container}`}
        role="alert"
        data-testid="infobar"
      >
        <Icon
          className={`size-5 shrink-0 ${variantStyles.icon}`}
          aria-hidden="true"
        />

        <div className="flex-1">
          <div className={`font-medium ${variantStyles.icon}`}>{title}</div>
          <div className="mt-0.5 text-sm text-muted-foreground">
            {description}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {actionLabel && onAction && (
            <Button
              onClick={onAction}
              variant="default"
              size="sm"
              data-testid="infobar-action"
            >
              {actionLabel}
            </Button>
          )}

          {onDismiss && (
            <Button
              onClick={onDismiss}
              variant="ghost"
              size="icon"
              aria-label={t("common.dismiss")}
            >
              <X className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
