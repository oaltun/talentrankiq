import * as React from "react"

import { NavMain } from "@/components/nav-main"
import { NavSecondary } from "@/components/nav-secondary"
import { SettingsDialog } from "@/components/settings/settings-dialog"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useI18n } from "@/i18n"
import {
  FolderCheckIcon,
  LayersIcon,
  Settings2Icon,
  CircleHelpIcon,
  CommandIcon,
} from "lucide-react"

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { t } = useI18n()
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false)

  const data = {
    navMain: [
      {
        title: t("nav.jobs"),
        url: "/jobs",
        sidebarDataTestId: "activity-home",
        icon: <LayersIcon />,
      },
      {
        title: t("nav.models"),
        url: "/models",
        sidebarDataTestId: "activity-models",
        icon: <FolderCheckIcon />,
      },
    ],
    navSecondary: [
      {
        title: t("nav.settings"),
        sidebarDataTestId: "sidebar-secondary-settings-btn",
        icon: <Settings2Icon />,
        onClick: () => setIsSettingsOpen(true),
      },
      {
        title: t("nav.help"),
        sidebarDataTestId: "sidebar-secondary-get-help-btn",
        icon: <CircleHelpIcon />,
      },
    ],
  }

  return (
    <>
      <Sidebar collapsible="offcanvas" {...props}>
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                data-testid="app-sidebar-icon-and-appname"
                className="data-[slot=sidebar-menu-button]:p-1.5!"
                render={<a href="#" />}
              >
                <CommandIcon className="size-5!" />
                <span
                  data-testid="app-sidebar-title"
                  className="text-base font-semibold"
                >
                  {t("app.name")}
                </span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        <SidebarContent>
          <NavMain items={data.navMain} />
          {/* <NavDocuments items={data.documents} /> */}
          <NavSecondary items={data.navSecondary} className="mt-auto" />
        </SidebarContent>
        {/* <SidebarFooter>
          <NavUser user={data.user} />
        </SidebarFooter> */}
      </Sidebar>
      <SettingsDialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen} />
    </>
  )
}
