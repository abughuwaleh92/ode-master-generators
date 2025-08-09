import React, { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { api } from './api/client'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'
import GeneratorPanel from './components/generation/GeneratorPanel'
import BatchGenerator from './components/generation/BatchGenerator'
import VerificationPanel from './components/verification/VerificationPanel'
import DatasetManager from './components/datasets/DatasetManager'
import MLDashboard from './components/ml/MLDashboard'
import SystemMonitor from './components/monitoring/SystemMonitor'
import JobsManager from './components/monitoring/JobsManager'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { isConnected } = useWebSocket()
  
  const { data: systemInfo } = useQuery('systemInfo', api.getInfo, {
    refetchInterval: 30000,
  })

  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        onMenuClick={() => setSidebarOpen(!sidebarOpen)}
        isConnected={isConnected}
        systemInfo={systemInfo}
      />
      
      <div className="flex">
        <Sidebar open={sidebarOpen} />
        
        <main className={`flex-1 transition-all duration-300 ${
          sidebarOpen ? 'ml-64' : 'ml-16'
        }`}>
          <div className="container mx-auto px-4 py-6">
            <Routes>
              <Route path="/" element={<Navigate to="/generate" />} />
              <Route path="/generate" element={<GeneratorPanel />} />
              <Route path="/batch" element={<BatchGenerator />} />
              <Route path="/verify" element={<VerificationPanel />} />
              <Route path="/datasets" element={<DatasetManager />} />
              <Route path="/ml" element={<MLDashboard />} />
              <Route path="/monitor" element={<SystemMonitor />} />
              <Route path="/jobs" element={<JobsManager />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
