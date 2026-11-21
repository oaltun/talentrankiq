import * as React from "react"

import {
  DEFAULT_LANGUAGE,
  LANGUAGES,
  LANGUAGE_STORAGE_KEY,
  translations,
  type Language,
  type TranslationVars,
} from "./translations"

type I18nContextValue = {
  language: Language
  setLanguage: (language: Language) => void
  t: (key: string, vars?: TranslationVars) => string
}

const I18nContext = React.createContext<I18nContextValue | undefined>(undefined)

function isLanguage(value: string | null): value is Language {
  return value !== null && (LANGUAGES as readonly string[]).includes(value)
}

function readInitialLanguage(): Language {
  if (typeof window === "undefined") {
    return DEFAULT_LANGUAGE
  }

  try {
    const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY)
    if (isLanguage(stored)) {
      return stored
    }
  } catch {
    // localStorage can be unavailable in hardened webviews; fall back.
  }

  return DEFAULT_LANGUAGE
}

function interpolate(template: string, vars?: TranslationVars): string {
  if (!vars) {
    return template
  }

  let result = template
  for (const [key, value] of Object.entries(vars)) {
    result = result.replaceAll(`{{${key}}}`, String(value))
  }
  return result
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] =
    React.useState<Language>(readInitialLanguage)

  React.useEffect(() => {
    document.documentElement.lang = language
  }, [language])

  const setLanguage = React.useCallback((next: Language) => {
    setLanguageState(next)
    try {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, next)
    } catch {
      // Persisting is best-effort; the in-memory switch still applies.
    }
  }, [])

  const t = React.useCallback(
    (key: string, vars?: TranslationVars) => {
      const entry = translations[key as keyof typeof translations]
      const template = entry?.[language] ?? entry?.[DEFAULT_LANGUAGE] ?? key
      return interpolate(template, vars)
    },
    [language]
  )

  const value = React.useMemo(
    () => ({ language, setLanguage, t }),
    [language, setLanguage, t]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const context = React.useContext(I18nContext)
  if (context === undefined) {
    throw new Error("useI18n must be used within an I18nProvider")
  }
  return context
}
