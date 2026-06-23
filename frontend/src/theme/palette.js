/**
 * Lively-turing warm palette (IU-DU MarketingPages + DashboardWorkspace).
 * Brown/espresso backgrounds, cream text, gold/orange accents — NOT blue storm.
 */
export const WARM = {
  bg: '#11100E',
  surface: '#1A1815',
  card: '#1A1B1E',
  dashboard: '#0B0B0C',
  900: '#2A241C',
  500: '#8F5C12',
  300: '#D69A2D',
  100: '#F4EBDD',
  muted: '#A69E92',
  ink: '#16130E',
}

/** Tailwind `storm-*` classes are remapped to these warm values in index.html */
export const STORM = WARM

export const SURFACE = {
  dashboardFrom: '#0B0B0C',
  dashboardVia: '#11100E',
  dashboardTo: '#1A1815',
  panel: 'rgba(26, 24, 21, 0.86)',
  panelBorder: 'rgba(244, 235, 221, 0.10)',
  glass: 'rgba(26, 24, 21, 0.86)',
  glassBorder: 'rgba(244, 235, 221, 0.10)',
}

export const ACCENT = {
  gold: '#D69A2D',
  goldHover: '#C8891E',
  bronze: '#8F5C12',
  amber: '#F59E0B',
  copper: '#B87333',
  cream: '#F4EBDD',
  muted: '#A69E92',
  terracotta: '#E07A5F',
  success: '#6B8F71',
}

export const GRADIENT = {
  cta: 'linear-gradient(90deg, #D69A2D, #C8891E)',
  ctaAlt: 'linear-gradient(90deg, #C8891E, #D69A2D)',
  headline: 'linear-gradient(90deg, #F4EBDD, #D69A2D, #8F5C12)',
  headlineAccent: 'linear-gradient(90deg, #D69A2D, #F59E0B)',
  logo: 'linear-gradient(135deg, #D69A2D, #8F5C12)',
  pipeline: 'linear-gradient(90deg, #D69A2D, #8F5C12)',
  spotlight: 'radial-gradient(540px circle at var(--spotlight-x, 50%) var(--spotlight-y, 20%), rgba(214,154,45,0.12), transparent 42%)',
}

export const WORKFLOW_ACCENTS = [
  { hex: ACCENT.gold, border: 'hover:border-[#D69A2D]/50', glow: 'from-[#D69A2D]/10', text: 'text-[#D69A2D]' },
  { hex: ACCENT.goldHover, border: 'hover:border-[#C8891E]/50', glow: 'from-[#C8891E]/10', text: 'text-[#C8891E]' },
  { hex: ACCENT.amber, border: 'hover:border-[#F59E0B]/50', glow: 'from-[#F59E0B]/10', text: 'text-[#F59E0B]' },
  { hex: ACCENT.bronze, border: 'hover:border-[#8F5C12]/50', glow: 'from-[#8F5C12]/10', text: 'text-[#8F5C12]' },
]

export const SEMANTIC = {
  danger: ACCENT.terracotta,
  warning: ACCENT.amber,
  success: ACCENT.success,
  info: ACCENT.gold,
  neutral: ACCENT.muted,
}

export function riskColor(pct) {
  if (pct >= 70) return ACCENT.terracotta
  if (pct >= 40) return ACCENT.amber
  return ACCENT.gold
}

export function stisColor(stis) {
  if (stis >= 7) return ACCENT.terracotta
  if (stis >= 4) return ACCENT.amber
  return ACCENT.success
}

export function cascadeColor(prob) {
  return prob > 0.3 ? ACCENT.terracotta : ACCENT.amber
}

export function accentBg(hex, alpha = 0.1) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}