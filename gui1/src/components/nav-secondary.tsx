"use client"

import * as React from "react"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useHashRouter } from "@/lib/hash-router"

export function NavSecondary({
  items,
  ...props
}: {
  items: {
    title: string
    url?: string
    icon: React.ReactNode
    sidebarDataTestId?: string
    onClick?: () => void
  }[]
} & React.ComponentPropsWithoutRef<typeof SidebarGroup>) {
  const { navigate } = useHashRouter()

  return (
    <SidebarGroup {...props}>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.title}>
              <SidebarMenuButton
                data-testid={item.sidebarDataTestId}
                tooltip={item.title}
                onClick={
                  item.onClick
                    ? item.onClick
                    : item.url
                      ? () => navigate(item.url ?? "")
                      : undefined
                }
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
