import type { CSSProperties } from "react"
import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import { JobsPage } from "@/components/pages/JobsPage"
import { ModelsPage } from "@/components/pages/ModelsPage"
import { JobDetailsPage } from "@/components/pages/JobDetailsPage"
import { HashRouter, useHashRouter } from "@/lib/hash-router"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"

function Content() {
  const { path } = useHashRouter()

  const renderPage = () => {
    if (path === "/jobs") {
      return <JobsPage />
    }
    if (path === "/models" || path.startsWith("/models?")) {
      return <ModelsPage />
    }
    if (path.startsWith("/job/")) {
      const jobId = path.split("/job/")[1]
      return <JobDetailsPage jobId={jobId ?? null} />
    }
    return <JobsPage />
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <SiteHeader />
      <div className="flex-1 overflow-auto p-4 lg:p-6">{renderPage()}</div>
    </div>
  )
}

export default function App() {
  return (
    <TooltipProvider>
      <HashRouter>
        <SidebarProvider
          style={
            {
              "--sidebar-width": "calc(var(--spacing) * 72)",
              "--header-height": "calc(var(--spacing) * 12)",
            } as CSSProperties
          }
        >
          <AppSidebar variant="inset" />
          <SidebarInset>
            <Content />
          </SidebarInset>
        </SidebarProvider>
      </HashRouter>
    </TooltipProvider>
  )
}
