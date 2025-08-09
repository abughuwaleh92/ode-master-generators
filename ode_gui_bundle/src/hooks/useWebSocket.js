import { useEffect, useState, useCallback, useRef } from 'react'
import { WS_ENABLED } from '../config'
import toast from 'react-hot-toast'

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const ws = useRef(null)
  const reconnectTimeout = useRef(null)
  const messageHandlers = useRef(new Map())

  const connect = useCallback(() => {
    if (!WS_ENABLED) return

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/${Date.now()}`
    
    try {
      ws.current = new WebSocket(wsUrl)

      ws.current.onopen = () => {
        setIsConnected(true)
        toast.success('WebSocket connected', { id: 'ws-connected' })
        
        // Subscribe to general updates
        ws.current.send(JSON.stringify({
          type: 'subscribe',
          topic: 'general'
        }))
      }

      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          setLastMessage(message)
          
          // Call registered handlers
          messageHandlers.current.forEach((handler) => {
            handler(message)
          })
        } catch (error) {
          console.error('WebSocket message parse error:', error)
        }
      }

      ws.current.onclose = () => {
        setIsConnected(false)
        
        // Attempt reconnect after 5 seconds
        reconnectTimeout.current = setTimeout(() => {
          connect()
        }, 5000)
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        toast.error('WebSocket connection error', { id: 'ws-error' })
      }
    } catch (error) {
      console.error('WebSocket connection failed:', error)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
    }
    
    if (ws.current) {
      ws.current.close()
      ws.current = null
    }
  }, [])

  const sendMessage = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    }
  }, [])

  const subscribe = useCallback((topic) => {
    sendMessage({ type: 'subscribe', topic })
  }, [sendMessage])

  const unsubscribe = useCallback((topic) => {
    sendMessage({ type: 'unsubscribe', topic })
  }, [sendMessage])

  const addMessageHandler = useCallback((id, handler) => {
    messageHandlers.current.set(id, handler)
    
    return () => {
      messageHandlers.current.delete(id)
    }
  }, [])

  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
    addMessageHandler,
  }
}
