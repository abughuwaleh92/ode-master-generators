// Runtime configuration from server
const config = window.ODE_CONFIG || {}

export const API_BASE = config.API_BASE || ''
export const API_KEY = config.API_KEY || localStorage.getItem('api_key') || ''
export const WS_ENABLED = config.WS !== false

// Save API key if provided
if (config.API_KEY && !localStorage.getItem('api_key')) {
  localStorage.setItem('api_key', config.API_KEY)
}

export const updateApiKey = (key) => {
  localStorage.setItem('api_key', key)
  window.location.reload()
}
