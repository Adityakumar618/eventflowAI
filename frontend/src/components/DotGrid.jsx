import { useEffect, useRef, useCallback } from 'react'

export default function DotGrid() {
  const canvasRef = useRef(null)
  const mouseRef = useRef({ x: -1000, y: -1000 })
  const animFrameRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let cols, rows
    const spacing = 40
    const baseRadius = 1.2
    const hoverRadius = 3.5
    const hoverRange = 120

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = document.querySelector('.landing-scroll')?.scrollHeight || window.innerHeight * 4
      cols = Math.ceil(canvas.width / spacing)
      rows = Math.ceil(canvas.height / spacing)
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const rect = canvas.getBoundingClientRect()
      let mx = -1000
      let my = -1000
      if (mouseRef.current.x !== -1000) {
        mx = mouseRef.current.x - rect.left
        my = mouseRef.current.y - rect.top
      }
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * spacing + spacing / 2
          const y = r * spacing + spacing / 2
          const dx = x - mx
          const dy = y - my
          const dist = Math.sqrt(dx * dx + dy * dy)
          const t = Math.max(0, 1 - dist / hoverRange)
          const radius = baseRadius + (hoverRadius - baseRadius) * t
          const alpha = 0.08 + 0.35 * t
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(214, 154, 45, ${alpha})`
          ctx.fill()
        }
      }
      animFrameRef.current = requestAnimationFrame(draw)
    }

    resize()
    draw()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [])

  const handleMouseMove = useCallback((e) => {
    mouseRef.current = { x: e.clientX, y: e.clientY }
  }, [])

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [handleMouseMove])

  return <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-0" />
}