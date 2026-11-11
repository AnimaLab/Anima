import { useEffect, useState } from 'react'

/** What the user chose. "system" follows the OS setting. */
export type ThemePreference = 'system' | 'light' | 'dark'
export type ResolvedTheme = 'light' | 'dark'

const STORAGE_KEY = 'anima_theme'
const CHANGE_EVENT = 'anima-theme-change'

function systemTheme(): ResolvedTheme {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function getThemePreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored
  } catch {
    // storage unavailable: fall through to the default
  }
  // The neon palette is designed dark-first, so dark is the default.
  return 'dark'
}

function resolve(preference: ThemePreference): ResolvedTheme {
  return preference === 'system' ? systemTheme() : preference
}

/** Apply a preference to <html> and remember it. */
export function setThemePreference(preference: ThemePreference) {
  try {
    localStorage.setItem(STORAGE_KEY, preference)
  } catch {
    // storage unavailable: the choice lasts for this page load
  }
  document.documentElement.dataset.theme = resolve(preference)
  window.dispatchEvent(new Event(CHANGE_EVENT))
}

/**
 * Current preference and resolved theme. Re-renders when the user switches
 * theme or, in "system" mode, when the OS setting changes.
 */
export function useTheme(): { preference: ThemePreference; theme: ResolvedTheme } {
  const [state, setState] = useState(() => {
    const preference = getThemePreference()
    return { preference, theme: resolve(preference) }
  })

  useEffect(() => {
    const sync = () => {
      const preference = getThemePreference()
      const theme = resolve(preference)
      document.documentElement.dataset.theme = theme
      setState({ preference, theme })
    }
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', sync)
    window.addEventListener(CHANGE_EVENT, sync)
    return () => {
      media.removeEventListener('change', sync)
      window.removeEventListener(CHANGE_EVENT, sync)
    }
  }, [])

  return state
}

/**
 * Read a design token for places CSS classes cannot reach (canvas drawing).
 * Pass the token name without the leading dashes, e.g. cssVar('color-ink').
 */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim()
}
