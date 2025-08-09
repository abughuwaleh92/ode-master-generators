import React from 'react'
import { Brain, TrendingUp, Zap, Database } from 'lucide-react'

export default function MLDashboard() {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-6">Machine Learning Pipeline</h2>
        
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-800">
            ML features are currently in beta. Full functionality will be available in the next release.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
            <Brain className="w-8 h-8 text-blue-600 mb-3" />
            <h3 className="font-semibold mb-1">Model Training</h3>
            <p className="text-sm text-gray-600">Train custom ODE generation models</p>
          </div>
          
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
            <TrendingUp className="w-8 h-8 text-green-600 mb-3" />
            <h3 className="font-semibold mb-1">Performance Metrics</h3>
            <p className="text-sm text-gray-600">Track model accuracy and efficiency</p>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
            <Zap className="w-8 h-8 text-purple-600 mb-3" />
            <h3 className="font-semibold mb-1">Novel Generation</h3>
            <p className="text-sm text-gray-600">Generate unique ODE patterns</p>
          </div>
          
          <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-6">
            <Database className="w-8 h-8 text-orange-600 mb-3" />
            <h3 className="font-semibold mb-1">Dataset Preparation</h3>
            <p className="text-sm text-gray-600">Prepare training datasets</p>
          </div>
        </div>
      </div>
    </div>
  )
}
