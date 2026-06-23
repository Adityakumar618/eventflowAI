import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Lock, Mail, Moon, Sun, TrafficCone } from 'lucide-react'
import { CursorSpotlight, DotGridLayer, FadeReveal, MagneticButton } from '../components/ui'
import { getTheme } from '../theme/theme'

export default function LoginPage({ setView, isLightMode, setIsLightMode }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const theme = getTheme(isLightMode)

  const handleSignIn = (e) => {
    e.preventDefault()
    setError('')
    if (!email) return setError('Please enter your email address.')
    if (!password || password.length < 6) return setError('Password must be at least 6 characters.')
    setView('dashboard')
  }

  return (
    <div className={`relative min-h-screen overflow-hidden font-sans ${theme.page}`}>
      <CursorSpotlight />
      <DotGridLayer theme={theme} opacity={0.6} />

      <div className="relative z-10 grid min-h-screen lg:grid-cols-[1.05fr_0.95fr]">
        <section className="hidden min-h-screen items-center px-10 py-16 lg:flex">
          <FadeReveal className="mx-auto max-w-xl">
            <button type="button" onClick={() => setView('landing')} className={`mb-12 inline-flex items-center gap-2 text-sm font-semibold ${theme.muted} transition-colors hover:text-[#D69A2D]`}>
              <ArrowLeft size={17} /> Back to Home
            </button>
            <div className={`mb-8 inline-flex items-center gap-2 rounded-full border ${theme.border} ${theme.surface} px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] ${theme.muted}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-[#D69A2D]" />
              BTP Command Center
            </div>
            <h1 className={`text-5xl font-semibold leading-[1.02] tracking-tight xl:text-6xl ${theme.text}`}>Operational Foresight for Every ASTraM Event</h1>
            <p className={`mt-6 text-lg leading-8 ${theme.muted}`}>Survival curves, STIS scoring, hotspot briefings, and multi-event triage in one workspace.</p>

            <div className={`mt-12 rounded-[28px] border ${theme.surface} p-6 backdrop-blur-xl`}>
              <div className="relative h-64 overflow-hidden rounded-2xl border border-[#D69A2D]/12 bg-[#11100E]">
                <svg className="absolute inset-0 h-full w-full" viewBox="0 0 520 280" aria-hidden="true">
                  {[
                    [260, 62, 146, 126], [260, 62, 374, 126], [146, 126, 214, 214],
                    [374, 126, 306, 214], [214, 214, 306, 214], [260, 62, 260, 154],
                  ].map(([x1, y1, x2, y2]) => (
                    <line key={`${x1}-${y1}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(214,154,45,.24)" strokeWidth="1.2" />
                  ))}
                  {[[260, 62, 56], [146, 126, 42], [374, 126, 42], [260, 154, 34]].map(([cx, cy, r], i) => (
                    <circle key={cx} cx={cx} cy={cy} r={r} fill={i === 0 ? 'rgba(214,154,45,.18)' : 'rgba(244,235,221,.06)'} stroke="rgba(214,154,45,.28)" />
                  ))}
                </svg>
                <div className="absolute inset-x-5 bottom-5 rounded-2xl border border-white/10 bg-[#1A1815]/90 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[#D69A2D]">Live intelligence</p>
                  <p className="mt-2 text-sm text-[#F4EBDD]">Hotspots, triage, and dossiers mapped for shift handover.</p>
                </div>
              </div>
            </div>
          </FadeReveal>
        </section>

        <section className="flex min-h-screen items-center justify-center px-5 py-8 sm:px-8">
          <FadeReveal className="w-full max-w-md">
            <div className="mb-8 flex items-center justify-between lg:hidden">
              <button type="button" onClick={() => setView('landing')} className={`inline-flex items-center gap-2 text-sm font-semibold ${theme.muted}`}>
                <ArrowLeft size={17} /> Home
              </button>
              <button type="button" onClick={() => setIsLightMode(!isLightMode)} className={`flex h-10 w-10 items-center justify-center rounded-xl border ${theme.border} ${theme.faint}`}>
                {isLightMode ? <Moon size={18} /> : <Sun size={18} />}
              </button>
            </div>

            <div className={`rounded-[30px] border ${theme.surface} p-7 shadow-2xl shadow-black/15 backdrop-blur-2xl sm:p-9`}>
              <div className="mb-8">
                <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl border border-[#D69A2D]/30 bg-[#D69A2D]/10 text-[#D69A2D]">
                  <TrafficCone size={24} />
                </div>
                <h2 className={`text-3xl font-semibold tracking-tight ${theme.text}`}>Welcome back</h2>
                <p className={`mt-2 text-sm leading-6 ${theme.muted}`}>Sign in to the EventFlow BTP command center.</p>
              </div>

              {error && (
                <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="mb-5 rounded-2xl border border-[#D69A2D]/30 bg-[#D69A2D]/10 px-4 py-3 text-sm text-[#D69A2D]">
                  {error}
                </motion.div>
              )}

              <form onSubmit={handleSignIn} className="space-y-4">
                <div>
                  <label className={`mb-2 block text-sm font-semibold ${theme.text}`}>Email</label>
                  <div className="relative">
                    <Mail className={`absolute left-4 top-1/2 -translate-y-1/2 ${theme.faint}`} size={18} />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className={`w-full rounded-2xl border ${theme.border} bg-transparent py-3.5 pl-12 pr-4 ${theme.text} outline-none transition-colors focus:border-[#D69A2D]/60`}
                      placeholder="officer@btp.gov.in"
                    />
                  </div>
                </div>
                <div>
                  <label className={`mb-2 block text-sm font-semibold ${theme.text}`}>Password</label>
                  <div className="relative">
                    <Lock className={`absolute left-4 top-1/2 -translate-y-1/2 ${theme.faint}`} size={18} />
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className={`w-full rounded-2xl border ${theme.border} bg-transparent py-3.5 pl-12 pr-4 ${theme.text} outline-none transition-colors focus:border-[#D69A2D]/60`}
                      placeholder="Password"
                    />
                  </div>
                </div>
                <MagneticButton
                  type="submit"
                  className="mt-2 flex w-full items-center justify-center gap-2 rounded-2xl bg-[#D69A2D] px-5 py-3.5 font-bold text-[#16130E] transition-colors hover:bg-[#C8891E]"
                >
                  Sign In <ArrowRight size={18} />
                </MagneticButton>
              </form>

              <button
                type="button"
                onClick={() => setIsLightMode(!isLightMode)}
                className={`mt-5 hidden w-full items-center justify-center gap-2 rounded-2xl border ${theme.border} py-3 text-sm font-semibold ${theme.muted} transition-colors hover:text-[#D69A2D] lg:flex`}
              >
                {isLightMode ? <Moon size={17} /> : <Sun size={17} />}
                {isLightMode ? 'Use dark mode' : 'Use light mode'}
              </button>
            </div>
          </FadeReveal>
        </section>
      </div>
    </div>
  )
}