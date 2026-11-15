import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { STATIC } from './api'

if (STATIC && !window.location.hash.startsWith('#/replay')) window.location.replace('#/replay')

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
