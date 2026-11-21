import { useState, useEffect } from "react"
import { useHashRouter } from "../../../lib/hash-router"
import { useI18n } from "../../../i18n"
import { ModelsView } from "./ModelsView"
import { ScoutModelCard } from "./ScoutModelCard"
import { InfoBar } from "../../layout"
import { Tabs, TabsList, TabsTrigger } from "../../ui/tabs"

const TABS = ["quick", "deep"] as const
type Tab = (typeof TABS)[number]

function parseTab(path: string): Tab {
  const q = path.split("?")[1]
  if (q) {
    const t = new URLSearchParams(q).get("tab")
    if (t && (TABS as readonly string[]).includes(t)) return t as Tab
  }
  const seg = path.split("/").filter(Boolean)
  if (
    seg.length >= 2 &&
    seg[0] === "models" &&
    (TABS as readonly string[]).includes(seg[1])
  ) {
    return seg[1] as Tab
  }
  return "deep"
}

function parseRedirectParams(path: string): {
  redirected: boolean
  reason: string | null
  model: string | null
} {
  const q = path.split("?")[1]
  if (!q) return { redirected: false, reason: null, model: null }
  const params = new URLSearchParams(q)
  const redirected = params.get("redirected_from_workspace") === "1"
  const reason = params.get("reason")
  const model = params.get("model")
  return { redirected, reason, model }
}

export function ModelsActivity() {
  const { path, navigate } = useHashRouter()
  const { t } = useI18n()
  const activeTab = parseTab(path)
  const { redirected, reason, model } = parseRedirectParams(path)
  const [showInfobar, setShowInfobar] = useState(redirected)

  useEffect(() => {
    setShowInfobar(redirected)
  }, [redirected])

  const handleDismiss = () => {
    setShowInfobar(false)
  }

  const handleAction = () => {
    // The workspace redirect tells us which model is missing; send the user to
    // that tab. Fall back to the tab currently shown.
    const target = reason === "quick" ? "quick" : "deep"
    navigate(`/models?tab=${target}`)
  }

  const modelLabel = model ? ` (${model})` : ""
  const description =
    reason === "quick"
      ? t("models.missing.description.quick")
      : reason === "default" || reason === "deep"
        ? t("models.missing.description.default", { model: modelLabel })
        : t("models.missing.description.generic")

  return (
    <>
      <InfoBar
        isVisible={showInfobar}
        variant="warning"
        title={t("models.missing.title")}
        description={description}
        actionLabel={t("models.missing.action")}
        onAction={handleAction}
        onDismiss={handleDismiss}
      />
      <div className="mb-4">
        <h1 className="text-lg font-semibold">{t("models.title")}</h1>
        <p className="text-sm text-muted-foreground">{t("models.subtitle")}</p>
      </div>
      <Tabs
        value={activeTab}
        onValueChange={(value) => navigate(`/models?tab=${value}`)}
      >
        <TabsList className="mb-4 gap-2">
          {TABS.map((tab) => (
            <TabsTrigger
              key={tab}
              value={tab}
              data-testid={`models-tab-${tab}`}
            >
              {t(`models.tab.${tab}`)}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      {activeTab === "quick" && (
        <div data-testid="models-content-quick">
          <ScoutModelCard />
        </div>
      )}
      {activeTab === "deep" && (
        <div data-testid="models-content-deep">
          <ModelsView />
        </div>
      )}
    </>
  )
}
