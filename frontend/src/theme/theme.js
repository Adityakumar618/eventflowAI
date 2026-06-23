/** Theme tokens from lively-turing IU-DU MarketingPages + DashboardWorkspace */

export const getTheme = (isLightMode) => ({
  page: isLightMode ? 'bg-[#F7F3EA] text-[#1E1D1A]' : 'bg-[#11100E] text-[#F4EBDD]',
  surface: isLightMode ? 'bg-white/86 border-[#1E1D1A]/10' : 'bg-[#1A1815]/86 border-[#F4EBDD]/10',
  surfaceSolid: isLightMode ? 'bg-white border-[#1E1D1A]/10' : 'bg-[#1A1815] border-[#F4EBDD]/10',
  text: isLightMode ? 'text-[#1E1D1A]' : 'text-[#F4EBDD]',
  muted: isLightMode ? 'text-[#6F6A60]' : 'text-[#A69E92]',
  faint: isLightMode ? 'text-[#1E1D1A]/50' : 'text-[#F4EBDD]/45',
  border: isLightMode ? 'border-[#1E1D1A]/10' : 'border-[#F4EBDD]/10',
  line: isLightMode ? 'bg-[#1E1D1A]/10' : 'bg-[#F4EBDD]/10',
  dotGlow: isLightMode ? '#F7F3EA' : '#11100E',
  dotFrom: isLightMode ? 'rgba(143, 92, 18, 0.16)' : 'rgba(214, 154, 45, 0.18)',
  dotTo: isLightMode ? 'rgba(143, 92, 18, 0.08)' : 'rgba(214, 154, 45, 0.08)',
})

export const getWorkspaceTheme = (isLightMode) => ({
  bg: isLightMode ? '#F7F3EA' : '#0B0B0C',
  surface: isLightMode ? '#FFFCF6' : '#131416',
  card: isLightMode ? '#FFFFFF' : '#1A1B1E',
  accent: isLightMode ? '#B45309' : '#F59E0B',
  text: isLightMode ? '#1E1D1A' : '#FAFAF9',
  secondary: isLightMode ? '#5C564C' : '#A1A1AA',
  border: isLightMode ? 'rgba(30,29,26,0.12)' : 'rgba(250,250,249,0.10)',
  softBorder: isLightMode ? 'rgba(30,29,26,0.08)' : 'rgba(250,250,249,0.07)',
  dotFrom: isLightMode ? 'rgba(180,83,9,0.14)' : 'rgba(245,158,11,0.14)',
  dotTo: isLightMode ? 'rgba(180,83,9,0.05)' : 'rgba(245,158,11,0.05)',
})