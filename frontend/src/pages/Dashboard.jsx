import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BarChart3, Calendar, LayoutDashboard, LogOut, MapPin, Moon, PanelLeftClose,
  PanelLeftOpen, Sun, TrafficCone, Zap,
} from 'lucide-react'
import { DotField } from '../components/DotField'
import NewEventView from '../views/NewEventView'
import BriefingView from '../views/BriefingView'
import AuditView from '../views/AuditView'
import PlannedView from '../views/PlannedView'
import { getWorkspaceTheme } from '../theme/theme'

const NAV = [
  { id: 'home', label: 'Overview', icon: LayoutDashboard },
  { id: 'new-event', label: 'New Event', icon: Zap },
  { id: 'briefing', label: 'Morning Briefing', icon: MapPin },
  { id: 'audit', label: 'Post-Event Audit', icon: BarChart3 },
  { id: 'planned', label: 'Planned Events', icon: Calendar },
]

function SidebarItem({ icon: Icon, label, collapsed, active, theme, onClick }) {
  return (
    <button
      type="button"
      title={collapsed ? label : undefined}
      onClick={onClick}
      className="group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-colors"
      style={{
        color: active ? theme.text : theme.secondary,
        background: active ? `${theme.accent}16` : 'transparent',
      }}
    >
      <Icon size={18} style={{ color: active ? theme.accent : theme.secondary }} />
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  )
}

function HomeOverview({ setTab, theme }) {
  const cards = [
    { id: 'new-event', title: 'New Event Intake', desc: 'Predict duration, STIS & triage for incoming ASTraM events.', icon: Zap },
    { id: 'briefing', title: 'Morning Briefing', desc: 'Hotspot heatmaps and risk zones for shift start.', icon: MapPin },
    { id: 'audit', title: 'Post-Event Audit', desc: 'Station efficiency and cascade chain validation.', icon: BarChart3 },
    { id: 'planned', title: 'Planned Events', desc: 'Pre-computed dossiers for processions & VIP routes.', icon: Calendar },
  ]
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold" style={{ color: theme.text }}>Command Center Overview</h2>
        <p className="mt-1 text-sm" style={{ color: theme.secondary }}>Gridlock Hackathon 2.0 · Bengaluru Traffic Police</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          ['24', 'Active Events'], ['4.2h', 'Avg Resolution'], ['7', 'High STIS Zones'], ['142', 'Officers Deployed'],
        ].map(([val, label]) => (
          <div key={label} className="rounded-2xl border p-4 text-center" style={{ borderColor: theme.softBorder, background: theme.card }}>
            <div className="text-2xl font-black" style={{ color: theme.accent }}>{val}</div>
            <div className="text-xs mt-1" style={{ color: theme.secondary }}>{label}</div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {cards.map((c) => (
          <button
            key={c.id}
            type="button"
            onClick={() => setTab(c.id)}
            className="group text-left rounded-2xl border p-6 transition-colors hover:border-[#D69A2D]/40"
            style={{ borderColor: theme.softBorder, background: theme.card }}
          >
            <c.icon size={28} style={{ color: theme.accent }} className="mb-3 group-hover:scale-110 transition-transform" />
            <h3 className="font-bold text-lg" style={{ color: theme.text }}>{c.title}</h3>
            <p className="text-sm mt-1" style={{ color: theme.secondary }}>{c.desc}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard({ setView, isLightMode, setIsLightMode }) {
  const theme = getWorkspaceTheme(isLightMode)
  const [tab, setTab] = useState('home')
  const [leftCollapsed, setLeftCollapsed] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', isLightMode)
    document.body.style.backgroundColor = isLightMode ? '#f7f3ea' : '#11100e'
    document.body.style.color = isLightMode ? '#1e1d1a' : '#f4ebdd'
  }, [isLightMode])

  const content = {
    home: <HomeOverview setTab={setTab} theme={theme} />,
    'new-event': <NewEventView />,
    briefing: <BriefingView />,
    audit: <AuditView />,
    planned: <PlannedView />,
  }

  const activeLabel = NAV.find((n) => n.id === tab)?.label || 'Overview'

  return (
    <div className="relative flex h-screen w-full overflow-hidden font-sans" style={{ background: theme.bg, color: theme.text }}>
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.12]">
        <DotField
          glowColor={theme.bg}
          gradientFrom={theme.dotFrom}
          gradientTo={theme.dotTo}
          dotRadius={1.35}
          dotSpacing={22}
          cursorRadius={300}
          cursorForce={0.03}
          bulgeStrength={18}
        />
      </div>

      <motion.aside
        animate={{ width: leftCollapsed ? 56 : 280 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 flex shrink-0 flex-col border-r"
        style={{ background: theme.surface, borderColor: theme.border }}
      >
        <div className="flex h-16 items-center justify-between px-3">
          <button type="button" onClick={() => setView('landing')} className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl" style={{ background: `${theme.accent}16`, color: theme.accent }}>
              <TrafficCone size={19} />
            </span>
            {!leftCollapsed && <span className="truncate text-sm font-semibold">EventFlow AI</span>}
          </button>
          {!leftCollapsed && (
            <button
              type="button"
              title="Collapse sidebar"
              onClick={() => setLeftCollapsed(true)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border"
              style={{ borderColor: theme.softBorder, color: theme.secondary }}
            >
              <PanelLeftClose size={17} />
            </button>
          )}
        </div>

        {leftCollapsed && (
          <button
            type="button"
            title="Expand sidebar"
            onClick={() => setLeftCollapsed(false)}
            className="mx-auto mb-3 flex h-9 w-9 items-center justify-center rounded-xl border"
            style={{ borderColor: theme.softBorder, color: theme.secondary }}
          >
            <PanelLeftOpen size={17} />
          </button>
        )}

        <div className="space-y-1 px-2 flex-1">
          {NAV.map((item) => (
            <SidebarItem
              key={item.id}
              icon={item.icon}
              label={item.label}
              collapsed={leftCollapsed}
              active={tab === item.id}
              theme={theme}
              onClick={() => setTab(item.id)}
            />
          ))}
        </div>

        <div className="space-y-1 px-2 pb-3 border-t pt-3" style={{ borderColor: theme.border }}>
          {!leftCollapsed && (
            <button
              type="button"
              onClick={() => setIsLightMode(!isLightMode)}
              className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium"
              style={{ color: theme.secondary }}
            >
              {isLightMode ? <Moon size={18} /> : <Sun size={18} />}
              {isLightMode ? 'Dark mode' : 'Light mode'}
            </button>
          )}
          <button
            type="button"
            onClick={() => setView('landing')}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium"
            style={{ color: theme.secondary }}
          >
            <LogOut size={18} />
            {!leftCollapsed && 'Exit'}
          </button>
        </div>
      </motion.aside>

      <main className="relative z-10 flex min-w-0 flex-1 flex-col">
        <header
          className="flex h-16 shrink-0 items-center justify-between border-b px-5"
          style={{ background: theme.surface, borderColor: theme.border }}
        >
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: theme.secondary }}>BTP Command</p>
            <h1 className="text-lg font-semibold">{activeLabel}</h1>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-5 md:p-6">
          <div className="mx-auto max-w-6xl rounded-3xl border p-5 md:p-6" style={{ borderColor: theme.softBorder, background: theme.card }}>
            <AnimatePresence mode="wait">
              <motion.div
                key={tab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="workspace-view"
                data-theme={isLightMode ? 'light' : 'dark'}
              >
                {content[tab]}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  )
}