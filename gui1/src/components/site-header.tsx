import { useHashRouter } from "@/lib/hash-router"
import { useI18n } from "@/i18n"
import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"

function pageNameKey(path: string): string {
  if (path.startsWith("/job/")) return "page.jobDetails"
  if (path.startsWith("/models")) return "page.models"
  if (path.startsWith("/jobs") || path.startsWith("/home")) return "page.jobs"
  return "page.jobs"
}

export function SiteHeader() {
  const { path } = useHashRouter()
  const { t } = useI18n()
  return (
    <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator
          orientation="vertical"
          className="mx-2 h-4 data-vertical:self-auto"
        />
        <span className="text-base font-medium">{t(pageNameKey(path))}</span>
      </div>
    </header>
  )
}
