import { motion } from 'framer-motion'

export default function FloatingOrb({ className = '', delay = 0 }) {
  return (
    <motion.div
      className={`absolute rounded-full pointer-events-none blur-3xl ${className}`}
      animate={{
        y: [0, -30, 0, 20, 0],
        x: [0, 15, -10, 5, 0],
        scale: [1, 1.1, 0.95, 1.05, 1],
      }}
      transition={{ duration: 12, repeat: Infinity, delay, ease: 'easeInOut' }}
    />
  )
}