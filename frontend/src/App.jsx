import { useEffect, useState } from 'react'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import Dashboard from './pages/Dashboard'

export default function App() {
  const [view, setView] = useState('landing')
  const [isLightMode, setIsLightMode] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', isLightMode)
    document.body.style.backgroundColor = isLightMode ? '#f7f3ea' : '#11100e'
    document.body.style.color = isLightMode ? '#1e1d1a' : '#f4ebdd'
  }, [isLightMode])

  if (view === 'landing') {
    return <LandingPage setView={setView} isLightMode={isLightMode} setIsLightMode={setIsLightMode} />
  }
  if (view === 'login') {
    return <LoginPage setView={setView} isLightMode={isLightMode} setIsLightMode={setIsLightMode} />
  }
  if (view === 'dashboard') {
    return <Dashboard setView={setView} isLightMode={isLightMode} setIsLightMode={setIsLightMode} />
  }
  return <LandingPage setView={setView} isLightMode={isLightMode} setIsLightMode={setIsLightMode} />
}