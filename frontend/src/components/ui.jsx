import { useEffect } from 'react'
import { motion, useMotionValue } from 'framer-motion'
import { DotField } from './DotField'

export function FadeReveal({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 22, filter: 'blur(10px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '-70px' }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

export function MagneticButton({ children, className = '', ...props }) {
  const x = useMotionValue(0)
  const y = useMotionValue(0)

  return (
    <motion.button
      style={{ x, y }}
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        x.set((e.clientX - rect.left - rect.width / 2) * 0.14)
        y.set((e.clientY - rect.top - rect.height / 2) * 0.14)
      }}
      onMouseLeave={() => { x.set(0); y.set(0) }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      className={className}
      {...props}
    >
      {children}
    </motion.button>
  )
}

export function CursorSpotlight() {
  useEffect(() => {
    const handleMouse = (e) => {
      document.documentElement.style.setProperty('--spotlight-x', `${e.clientX}px`)
      document.documentElement.style.setProperty('--spotlight-y', `${e.clientY}px`)
    }
    window.addEventListener('mousemove', handleMouse, { passive: true })
    return () => window.removeEventListener('mousemove', handleMouse)
  }, [])

  return (
    <div
      className="pointer-events-none fixed inset-0 z-[1] opacity-70"
      style={{
        background: 'radial-gradient(540px circle at var(--spotlight-x, 50%) var(--spotlight-y, 20%), rgba(214,154,45,0.12), transparent 42%)',
      }}
    />
  )
}

export function DotGridLayer({ theme, opacity = 1 }) {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden" style={{ opacity }}>
      <DotField
        glowColor={theme.dotGlow}
        gradientFrom={theme.dotFrom}
        gradientTo={theme.dotTo}
        dotRadius={1.4}
        dotSpacing={20}
        cursorRadius={360}
        cursorForce={0.045}
        bulgeStrength={26}
      />
    </div>
  )
}