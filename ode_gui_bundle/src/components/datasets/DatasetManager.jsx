import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { api } from '../../api/client'
import { Database, Download, Trash2, Eye, Plus, Search } from 'lucide-react'
import toast from 'react-hot-toast'
import { format } from 'date-fns'
import DatasetViewer from './DatasetViewer'

export default function DatasetManager() {
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [showViewer, setShowViewer] = useState(false)
  const queryClient = useQueryClient()

  const { data: datasets, isLoading } = useQuery('datasets', api.listDatasets, {
    refetchInterval: 30000,
  })

  const downloadDataset = async (name, format) => {
    try {
      const response = await api.downloadDataset(name, format)
      const url = window.URL.createObjectURL(response.data)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.${format}`
      a.click()
      toast.success(`Downloaded ${name}.${format}`)
    } catch (error) {
      toast.error('Download failed')
    }
  }

  const filteredDatasets = datasets?.datasets?.filter(ds =>
    ds.name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || []

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Dataset Manager</h2>
          
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search datasets..."
                className="pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <button className="btn-primary flex items-center">
              <Plus className="w-5 h-5 mr-2" />
              Create Dataset
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-8">Loading datasets...</div>
        ) : filteredDatasets.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Database className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>No datasets found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    ODEs
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Verification Rate
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredDatasets.map((dataset) => (
                  <tr key={dataset.name} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Database className="w-5 h-5 text-gray-400 mr-2" />
                        <span className="text-sm font-medium text-gray-900">
                          {dataset.name}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatBytes(dataset.size_bytes)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {dataset.metadata?.size || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {dataset.metadata?.verification_rate ? (
                        <div className="flex items-center">
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                            <div
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${dataset.metadata.verification_rate * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-600">
                            {(dataset.metadata.verification_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {format(new Date(dataset.created_at), 'MMM d, yyyy')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => {
                            setSelectedDataset(dataset)
                            setShowViewer(true)
                          }}
                          className="text-blue-600 hover:text-blue-900"
                          title="View"
                        >
                          <Eye className="w-5 h-5" />
                        </button>
                        
                        <div className="relative group">
                          <button
                            className="text-green-600 hover:text-green-900"
                            title="Download"
                          >
                            <Download className="w-5 h-5" />
                          </button>
                          
                          <div className="absolute right-0 mt-2 w-32 bg-white rounded-md shadow-lg z-10 hidden group-hover:block">
                            <button
                              onClick={() => downloadDataset(dataset.name, 'jsonl')}
                              className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                            >
                              JSONL
                            </button>
                            <button
                              onClick={() => downloadDataset(dataset.name, 'csv')}
                              className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                            >
                              CSV
                            </button>
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Dataset Viewer Modal */}
      {showViewer && selectedDataset && (
        <DatasetViewer
          dataset={selectedDataset}
          onClose={() => {
            setShowViewer(false)
            setSelectedDataset(null)
          }}
        />
      )}

      {/* Dataset Statistics */}
      {datasets?.datasets?.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Dataset Statistics</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {datasets.count}
              </div>
              <div className="text-sm text-gray-600">Total Datasets</div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {datasets.datasets.reduce((sum, ds) => 
                  sum + (ds.metadata?.size || 0), 0
                ).toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Total ODEs</div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {formatBytes(
                  datasets.datasets.reduce((sum, ds) => sum + ds.size_bytes, 0)
                )}
              </div>
              <div className="text-sm text-gray-600">Total Size</div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-gray-900">
                {(
                  datasets.datasets.reduce((sum, ds) => 
                    sum + (ds.metadata?.verification_rate || 0), 0
                  ) / datasets.count * 100
                ).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Avg Verification</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
