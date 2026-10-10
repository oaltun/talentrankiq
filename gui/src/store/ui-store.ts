import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface UIState {
  commandPaletteOpen: boolean
  preferencesOpen: boolean
  lastQuickPaneEntry: string | null
  currentJobId: string | null
  /** Job ids whose "add job ad" nudge banner the user dismissed this session
   * (transient view state — the banner itself is backend-derived). */
  dismissedNudges: string[]

  toggleCommandPalette: () => void
  setCommandPaletteOpen: (open: boolean) => void
  togglePreferences: () => void
  setPreferencesOpen: (open: boolean) => void
  setLastQuickPaneEntry: (text: string) => void
  setSquareCorners: (enabled: boolean) => void
  setCurrentJobId: (id: string | null) => void
  dismissNudge: (jobId: string) => void
}

export const useUIStore = create<UIState>()(
  devtools(
    set => ({
      commandPaletteOpen: false,
      preferencesOpen: false,
      lastQuickPaneEntry: null,
      currentJobId: null,
      dismissedNudges: [],

      toggleCommandPalette: () =>
        set(
          state => ({ commandPaletteOpen: !state.commandPaletteOpen }),
          undefined,
          'toggleCommandPalette'
        ),

      setCommandPaletteOpen: open =>
        set({ commandPaletteOpen: open }, undefined, 'setCommandPaletteOpen'),

      togglePreferences: () =>
        set(
          state => ({ preferencesOpen: !state.preferencesOpen }),
          undefined,
          'togglePreferences'
        ),

      setPreferencesOpen: open =>
        set({ preferencesOpen: open }, undefined, 'setPreferencesOpen'),

      setLastQuickPaneEntry: text =>
        set({ lastQuickPaneEntry: text }, undefined, 'setLastQuickPaneEntry'),

      setSquareCorners: (enabled: boolean) => {
        document.documentElement.classList.toggle('square-corners', enabled)
      },

      setCurrentJobId: id =>
        set({ currentJobId: id }, undefined, 'setCurrentJobId'),

      dismissNudge: jobId =>
        set(
          state => ({
            dismissedNudges: state.dismissedNudges.includes(jobId)
              ? state.dismissedNudges
              : [...state.dismissedNudges, jobId],
          }),
          undefined,
          'dismissNudge'
        ),
    }),
    {
      name: 'ui-store',
    }
  )
)
