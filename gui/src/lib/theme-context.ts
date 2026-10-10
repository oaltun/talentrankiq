import { createContext } from 'react'

export type Theme = 'dark' | 'light' | 'system'

/** Color schemes (palettes) layered on top of light/dark mode. */
export const COLOR_THEMES = [
  'default',
  'slate',
  'ocean',
  'forest',
  'amber',
  'contrast',
] as const
export type ColorTheme = (typeof COLOR_THEMES)[number]

export interface ThemeProviderState {
  theme: Theme
  setTheme: (theme: Theme) => void
  colorTheme: ColorTheme
  setColorTheme: (theme: ColorTheme) => void
}

const initialState: ThemeProviderState = {
  theme: 'system',
  setTheme: () => null,
  colorTheme: 'default',
  setColorTheme: () => null,
}

export const ThemeProviderContext =
  createContext<ThemeProviderState>(initialState)
