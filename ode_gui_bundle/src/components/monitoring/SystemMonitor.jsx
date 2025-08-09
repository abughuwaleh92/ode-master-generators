import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { api } from '../../api/client'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { Activity, Cpu, HardDrive, Zap, AlertCircle } from 'lucide-react'

export default function SystemMonitor() {
  const [metrics, setMetrics] = useState({
    cpu: [],
    memory: [],
    generation_rate: [],
    verification_rate: [],
  })

  const { data: systemInfo } = useQuery('systemInfo', api.getInfo, {
    refetchInterval: 5000,
  })

  const { data: healthData } = useQuery('health', api.getHealth, {
    refetchInterval: 10000,
  })

  // Simulate real-time metrics (in production, use WebSocket)
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString()
      
      setMetrics(prev => ({
        cpu: [...prev.cpu.slice(-19), {
          time: now,
          value: 20 + Math.random() * 60,
        }],
        memory: [...prev.memory.slice(-19), {
          time: now,
          value: 40 + Math.random() * 40,
        }],
        generation_rate: [...prev.generation_rate.slice(-19), {
          time: now,
          value: Math.floor(Math.random() * 100),
        }],
        verification_rate: [...prev.verification_rate.slice(-19), {
          time: now,
          value: 70 + Math.random() * 30,
        }],
      }))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const serviceStatus = systemInfo?.services || {}
  const queueStats = systemInfo?.queues || {}

  const statusColor = (status) => {
    if (status === true || status === 'ok') return 'text-green-500'
    if (status === false) return 'text-red-500'
    return 'text-yellow-500'
  }

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">System Health</p>
              <p className={`text-2xl font-bold ${healthData === 'ok' ? 'text-green-500' : 'text-red-500'}`}>
                {healthData === 'ok' ? 'Healthy' : 'Issues'}
              </p>
            </div>
            <Activity className={`w-8 h-8 ${healthData === 'ok' ? 'text-green-500' : 'text-red-500'}`} />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">CPU Usage</p>
              <p className="text-2xl font-bold">
                {metrics.cpu[metrics.cpu.length - 1]?.value.toFixed(1) || 0}%
              </p>
            </div>
            <Cpu className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Memory Usage</p>
              <p className="text-2xl font-bold">
                {metrics.memory[metrics.memory.length - 1]?.value.toFixed(1) || 0}%
              </p>
            </div>
            <HardDrive className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Generation Rate</p>
              <p className="text-2xl font-bold">
                {metrics.generation_rate[metrics.generation_rate.length - 1]?.value || 0}/min
              </p>
            </div>
            <Zap className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
      </div>

      {/* Service Status */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Service Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(serviceStatus).map(([service, status]) => (
            <div key={service} className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${status ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="capitalize">{service}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">CPU & Memory Usage</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics.cpu}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                name="CPU %"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Generation Metrics</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics.generation_rate}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#10B981" 
                name="ODEs/min"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Queue Status */}
      {Object.keys(queueStats).length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Queue Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(queueStats).map(([queue, count]) => (
              <div key={queue} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="font-medium capitalize">{queue}</span>
                <span className="text-2xl font-bold text-blue-600">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
