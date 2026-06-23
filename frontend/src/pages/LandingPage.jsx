import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ArrowRight, BarChart3, Calendar, CheckCircle2, MapPin, Moon, Shield,
  Sparkles, Sun, TrafficCone, Zap,
} from 'lucide-react'
import SpotlightCard from '../components/SpotlightCard'
import { CursorSpotlight, DotGridLayer, FadeReveal, MagneticButton } from '../components/ui'
import { getTheme } from '../theme/theme'

const workflows = [
  { icon: Zap, title: 'New Event Intake', desc: 'Survival curves, STIS scoring, economic impact & triage.' },
  { icon: MapPin, title: 'Morning Briefing', desc: 'Hotspot heatmaps and risk zones before shift handover.' },
  { icon: BarChart3, title: 'Post-Event Audit', desc: 'Station efficiency, model validation, cascade analysis.' },
  { icon: Calendar, title: 'Planned Events', desc: '24-hour pre-event dossiers for processions & VIP routes.' },
]

const pipeline = [
  { label: 'Ingest', icon: Sparkles },
  { label: 'Predict', icon: Zap },
  { label: 'Score', icon: BarChart3 },
  { label: 'Triage', icon: MapPin },
  { label: 'Deploy', icon: Shield },
]

function CommandPreview({ theme }) {
  return (
    <SpotlightCard className={`rounded-[28px] ${theme.surface} shadow-2xl shadow-black/20`}>
      <div className="relative p-4 sm:p-5">
        <div className={`mb-4 flex items-center justify-between border-b ${theme.border} pb-4`}>
          <div>
            <p className={`text-xs font-semibold uppercase tracking-[0.24em] ${theme.faint}`}>Command Center</p>
            <p className={`mt-1 text-sm font-semibold ${theme.text}`}>BTP Bengaluru · Live</p>
          </div>
          <div className="rounded-full border border-[#D69A2D]/25 bg-[#D69A2D]/10 px-3 py-1 text-xs font-semibold text-[#D69A2D]">
            GridGuard V9
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
          <div className={`rounded-2xl border ${theme.border} bg-black/[0.03] p-4`}>
            <p className={`mb-4 flex items-center gap-2 text-sm font-semibold ${theme.text}`}>
              <Zap size={17} className="text-[#D69A2D]" />
              Active workflows
            </p>
            {['Water logging · Hebbal', 'VIP route · MG Road', 'Procession · Koramangala'].map((item, index) => (
              <motion.div
                key={item}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + index * 0.12 }}
                className={`mb-3 rounded-xl border ${theme.border} px-3 py-3`}
              >
                <div className={`text-sm font-medium ${theme.text}`}>{item}</div>
                <div className="mt-2 h-1.5 rounded-full bg-[#D69A2D]/12">
                  <motion.div
                    className="h-full rounded-full bg-[#D69A2D]"
                    initial={{ width: '18%' }}
                    animate={{ width: `${68 + index * 10}%` }}
                    transition={{ duration: 1.1, delay: 0.5 + index * 0.15 }}
                  />
                </div>
              </motion.div>
            ))}
          </div>

          <div className={`relative min-h-[310px] overflow-hidden rounded-2xl border ${theme.border} bg-[#11100E] p-5`}>
            <svg className="absolute inset-0 h-full w-full" viewBox="0 0 420 320" aria-hidden="true">
              {[
                ['210,82', '106,150'], ['210,82', '306,146'], ['106,150', '166,236'],
                ['306,146', '248,238'], ['166,236', '248,238'], ['210,82', '210,170'],
              ].map(([a, b]) => {
                const [x1, y1] = a.split(',')
                const [x2, y2] = b.split(',')
                return <line key={`${a}-${b}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(214,154,45,.28)" strokeWidth="1.2" />
              })}
            </svg>
            {[
              { label: 'STIS 8.2', x: '47%', y: '20%' },
              { label: '4.2h', x: '20%', y: '43%' },
              { label: 'Zone E', x: '73%', y: '42%' },
              { label: 'Triage', x: '34%', y: '72%' },
              { label: 'Deploy', x: '59%', y: '73%' },
            ].map((node, index) => (
              <motion.div
                key={node.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.55 + index * 0.12 }}
                className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#D69A2D]/35 bg-[#1A1815] px-3 py-2 text-xs font-semibold text-[#F4EBDD] shadow-[0_0_0_6px_rgba(214,154,45,0.06)]"
                style={{ left: node.x, top: node.y }}
              >
                {node.label}
              </motion.div>
            ))}
            <div className="absolute bottom-4 left-4 right-4 rounded-2xl border border-[#F4EBDD]/10 bg-[#1A1815]/92 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#D69A2D]">Triage alert</p>
              <p className="mt-2 text-sm text-[#F4EBDD]">3 concurrent events in Central zone — 14 officers available</p>
              <div className="mt-3 flex items-center gap-2 text-xs text-[#A69E92]">
                <CheckCircle2 size={14} className="text-[#D69A2D]" />
                Survival + STIS intelligence ready
              </div>
            </div>
          </div>
        </div>
      </div>
    </SpotlightCard>
  )
}

export default function LandingPage({ setView, isLightMode, setIsLightMode }) {
  const [scrollY, setScrollY] = useState(0)
  const theme = getTheme(isLightMode)

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', isLightMode)
  }, [isLightMode])

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className={`landing-scroll relative min-h-screen overflow-x-hidden font-sans ${theme.page}`}>
      <CursorSpotlight />

      <motion.nav
        initial={{ y: -24, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="fixed left-0 right-0 top-0 z-50 px-4 py-4"
      >
        <div className={`mx-auto flex max-w-7xl items-center justify-between rounded-2xl border px-4 py-3 backdrop-blur-xl transition-all duration-300 ${scrollY > 24 ? `${theme.surface} shadow-xl shadow-black/10` : 'border-transparent bg-transparent'}`}>
          <button type="button" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className={`flex items-center gap-3 ${theme.text}`}>
            <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-[#D69A2D]/30 bg-[#D69A2D]/12">
              <TrafficCone size={19} className="text-[#D69A2D]" />
            </span>
            <span className="text-lg font-bold tracking-tight">EventFlow AI</span>
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsLightMode(!isLightMode)}
              className={`flex h-10 w-10 items-center justify-center rounded-xl border ${theme.border} ${theme.faint} transition-colors hover:text-[#D69A2D]`}
              title="Toggle theme"
            >
              {isLightMode ? <Moon size={18} /> : <Sun size={18} />}
            </button>
            <button type="button" onClick={() => setView('login')} className={`hidden rounded-xl px-4 py-2 text-sm font-semibold transition-colors hover:text-[#D69A2D] sm:block ${theme.muted}`}>
              Sign In
            </button>
            <MagneticButton
              type="button"
              onClick={() => setView('dashboard')}
              className="rounded-xl bg-[#D69A2D] px-5 py-2.5 text-sm font-bold text-[#16130E] shadow-lg shadow-[#D69A2D]/15 transition-colors hover:bg-[#C8891E]"
            >
              Open Command Center
            </MagneticButton>
          </div>
        </div>
      </motion.nav>

      <section className="relative min-h-screen overflow-hidden px-6 pb-24 pt-32">
        <DotGridLayer theme={theme} opacity={1} />
        <div className="relative z-10 mx-auto grid max-w-7xl items-center gap-12 lg:min-h-[calc(100vh-8rem)] lg:grid-cols-[0.92fr_1.08fr]">
          <FadeReveal className="max-w-3xl">
            <p className="text-sm font-bold uppercase tracking-[0.24em] text-[#D69A2D]">Gridlock Hackathon 2.0</p>
            <h1 className={`mt-4 max-w-4xl text-5xl font-semibold leading-[0.98] tracking-tight sm:text-6xl lg:text-7xl ${theme.text}`}>
              Predict Impact. Deploy Smarter.
            </h1>
            <p className={`mt-7 max-w-2xl text-lg leading-8 sm:text-xl ${theme.muted}`}>
              Survival analysis, STIS scoring, and multi-event triage — operational foresight for Bengaluru Traffic Police before and during ASTraM events.
            </p>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <MagneticButton
                type="button"
                onClick={() => setView('dashboard')}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-[#D69A2D] px-7 py-4 text-base font-bold text-[#16130E] shadow-xl shadow-[#D69A2D]/18 transition-colors hover:bg-[#C8891E]"
              >
                Launch Command Center <ArrowRight size={19} />
              </MagneticButton>
              <MagneticButton
                type="button"
                onClick={() => setView('login')}
                className={`inline-flex items-center justify-center gap-2 rounded-2xl border ${theme.border} px-7 py-4 text-base font-semibold ${theme.text} transition-colors hover:border-[#D69A2D]/50 hover:text-[#D69A2D]`}
              >
                Officer Sign In
              </MagneticButton>
            </div>
            <div className={`mt-10 flex flex-wrap gap-x-6 gap-y-2 text-sm ${theme.faint}`}>
              <span className="flex items-center gap-2"><Shield size={15} className="text-[#D69A2D]" /> ASTraM Data</span>
              <span className="flex items-center gap-2"><Zap size={15} className="text-[#D69A2D]" /> GridGuard Inference</span>
              <span className="flex items-center gap-2"><MapPin size={15} className="text-[#D69A2D]" /> Real-time Triage</span>
            </div>
          </FadeReveal>
          <FadeReveal delay={0.12}>
            <CommandPreview theme={theme} />
          </FadeReveal>
        </div>
      </section>

      <section className="relative z-10 px-6 py-28">
        <div className="mx-auto max-w-6xl">
          <FadeReveal className="mx-auto max-w-3xl text-center">
            <p className="text-sm font-bold uppercase tracking-[0.24em] text-[#D69A2D]">Pipeline</p>
            <h2 className={`mt-4 text-3xl font-semibold tracking-tight sm:text-5xl ${theme.text}`}>From incoming event to deployment decision.</h2>
          </FadeReveal>
          <div className="mt-16 grid gap-4 md:grid-cols-5">
            {pipeline.map((step, index) => (
              <FadeReveal key={step.label} delay={index * 0.06} className="relative">
                <div className={`relative h-full rounded-2xl border ${theme.surfaceSolid} p-5 text-center shadow-sm`}>
                  <div className="mx-auto flex h-13 w-13 items-center justify-center rounded-2xl border border-[#D69A2D]/24 bg-[#D69A2D]/10 text-[#D69A2D]">
                    <step.icon size={24} />
                  </div>
                  <p className={`mt-4 text-lg font-semibold ${theme.text}`}>{step.label}</p>
                </div>
              </FadeReveal>
            ))}
          </div>
        </div>
      </section>

      <section className="relative z-10 px-6 py-28">
        <DotGridLayer theme={theme} opacity={0.18} />
        <div className="relative z-10 mx-auto max-w-6xl">
          <FadeReveal className="mb-14 max-w-3xl">
            <p className="text-sm font-bold uppercase tracking-[0.24em] text-[#D69A2D]">Workflows</p>
            <h2 className={`mt-4 text-3xl font-semibold tracking-tight sm:text-5xl ${theme.text}`}>Four command center modules for BTP operations.</h2>
          </FadeReveal>
          <div className="grid gap-5 md:grid-cols-2">
            {workflows.map((w, index) => (
              <FadeReveal key={w.title} delay={index * 0.06}>
                <motion.button
                  type="button"
                  whileHover={{ y: -5 }}
                  onClick={() => setView('dashboard')}
                  className={`group w-full text-left min-h-[230px] rounded-3xl border ${theme.surfaceSolid} p-8 transition-colors hover:border-[#D69A2D]/50`}
                >
                  <div className="mb-8 flex h-12 w-12 items-center justify-center rounded-2xl border border-[#D69A2D]/25 bg-[#D69A2D]/10 text-[#D69A2D]">
                    <w.icon size={24} />
                  </div>
                  <h3 className={`text-2xl font-semibold tracking-tight ${theme.text}`}>{w.title}</h3>
                  <p className={`mt-4 max-w-xl text-base leading-7 ${theme.muted}`}>{w.desc}</p>
                </motion.button>
              </FadeReveal>
            ))}
          </div>
        </div>
      </section>

      <section className="relative overflow-hidden px-6 py-32">
        <DotGridLayer theme={theme} opacity={0.95} />
        <div className="absolute left-1/2 top-1/2 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#D69A2D]/12 blur-[90px]" />
        <FadeReveal className="relative z-10 mx-auto max-w-4xl text-center">
          <h2 className={`text-4xl font-semibold tracking-tight sm:text-6xl ${theme.text}`}>Ready to command Bengaluru traffic?</h2>
          <p className={`mx-auto mt-6 max-w-2xl text-lg leading-8 ${theme.muted}`}>Give every officer operational foresight before congestion becomes gridlock.</p>
          <MagneticButton
            type="button"
            onClick={() => setView('dashboard')}
            className="mt-10 inline-flex items-center justify-center gap-2 rounded-2xl bg-[#D69A2D] px-8 py-4 text-base font-bold text-[#16130E] shadow-2xl shadow-[#D69A2D]/20 transition-colors hover:bg-[#C8891E]"
          >
            Launch Command Center <ArrowRight size={19} />
          </MagneticButton>
        </FadeReveal>
      </section>

      <footer className={`relative z-10 border-t px-6 py-10 ${theme.border}`}>
        <div className={`mx-auto flex max-w-7xl flex-col gap-5 text-sm md:flex-row md:items-center md:justify-between ${theme.muted}`}>
          <div className={`flex items-center gap-3 font-semibold ${theme.text}`}>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-[#D69A2D]/30 bg-[#D69A2D]/10">
              <TrafficCone size={17} className="text-[#D69A2D]" />
            </span>
            EventFlow AI
          </div>
          <span>2026 · Bengaluru Traffic Police · React + FastAPI</span>
        </div>
      </footer>
    </div>
  )
}