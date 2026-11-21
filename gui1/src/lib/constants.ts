import type { LucideIcon } from "lucide-react"
import { Home, Boxes } from "lucide-react"

export const GUI_PROTOCOL_VERSION = "1.0.0"
export const KNOWN_PROTOCOL_MAJOR = 1

export type ActivityKey = "home" | "models"

export interface Activity {
  key: ActivityKey
  label: string
  icon: LucideIcon
  path: string
}

export const ACTIVITIES: Activity[] = [
  { key: "home", label: "Home", icon: Home, path: "/home" },
  { key: "models", label: "Models", icon: Boxes, path: "/models" },
]
