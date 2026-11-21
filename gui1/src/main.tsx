import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./App.tsx"
import { I18nProvider } from "@/i18n"
import { ThemeProvider } from "@/components/theme-provider.tsx"
import { ExternalLinkGuard } from "./components/external-link-guard.tsx"
import { DebugPanel } from "./components/debug-panel.tsx"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <ThemeProvider>
        <ExternalLinkGuard />
        {import.meta.env.DEV ? <DebugPanel /> : null}
        <main data-ui-scroll-container>
          <App />
        </main>
      </ThemeProvider>
    </I18nProvider>
  </StrictMode>
)
