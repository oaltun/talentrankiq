import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useHashRouter } from "@/lib/hash-router"

export function NavMain({
  items,
}: {
  items: {
    title: string
    url: string
    sidebarDataTestId?: string
    icon?: React.ReactNode
  }[]
}) {
  const { navigate, path } = useHashRouter()

  const handleNavigate = (url: string) => {
    navigate(url)
  }

  const isActive = (url: string) => {
    return (
      path === url || path.startsWith(url + "/") || path.startsWith(url + "?")
    )
  }

  return (
    <SidebarGroup>
      <SidebarGroupContent className="flex flex-col gap-2">
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.title}>
              <SidebarMenuButton
                data-testid={item.sidebarDataTestId}
                tooltip={item.title}
                onClick={() => handleNavigate(item.url)}
                className={`data-[state=active]:bg-accent data-[state=active]:text-accent-foreground ${isActive(item.url) ? "data-[state=active]" : ""}`}
                aria-current={isActive(item.url) ? "page" : undefined}
              >
                {item.icon}
                <span>{item.title}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
