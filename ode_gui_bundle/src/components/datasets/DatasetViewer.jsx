import React, { useState } from 'react'
import { X, Search, Filter, Download } from 'lucide-react'
import ODEDisplay from '../common/ODEDisplay'

export default function DatasetViewer({ dataset, onClose }) {
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(10)
  
  // Mock data - in production, fetch from API
  const [odes] = useState([
    {
      id: 1,
      generator: 'L1',
      function: 'sine',
      ode: 'y\'\'(x) + y(x) = sin(x)',
      solution: 'sin(x)',
      verified: true,
      parameters: { alpha: 1, beta: 1, M: 0 },
    },
    // Add more mock data as needed
  ])

  const filteredOdes = odes.filter(ode =>
    JSON.stringify(ode).toLowerCase().includes(searchTerm.toLowerCase())
  )

  const totalPages = Math.ceil(filteredOdes.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const displayedOdes = filteredOdes.slice(startIndex, startIndex + itemsPerPage)

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b">
          <div>
            <h2 className="text-2xl font-bold">{dataset.name}</h2>
            <p className="text-sm text-gray-600 mt-1">
              {dataset.metadata?.size || 0} ODEs • 
              {dataset.metadata?.verification_rate 
                ? ` ${(dataset.metadata.verification_rate * 100).toFixed(1)}% verified`
                : ' No verification data'
              }
            </p>
          </div>
          
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Search Bar */}
        <div className="p-4 border-b">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search ODEs..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-4">
            {displayedOdes.map((ode, idx) => (
              <ODEDisplay 
                key={ode.id || idx} 
                ode={ode} 
                index={startIndex + idx + 1} 
              />
            ))}
          </div>
        </div>

        {/* Pagination */}
        <div className="border-t p-4 flex justify-between items-center">
          <div className="text-sm text-gray-600">
            Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, filteredOdes.length)} of {filteredOdes.length}
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 border rounded disabled:opacity-50"
            >
              Previous
            </button>
            
            <span className="px-3 py-1">
              Page {currentPage} of {totalPages}
            </span>
            
            <button
              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 border rounded disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
