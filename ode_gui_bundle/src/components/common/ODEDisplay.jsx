import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Check, X, Copy } from 'lucide-react'
import LatexRenderer from './LatexRenderer'
import toast from 'react-hot-toast'

export default function ODEDisplay({ ode, index }) {
  const [expanded, setExpanded] = useState(false)

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }

  return (
    <div className="border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-sm font-semibold text-gray-600">
              #{index}
            </span>
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
              {ode.generator}
            </span>
            <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">
              {ode.function}
            </span>
            {ode.verified !== undefined && (
              <span className={`flex items-center text-xs ${ode.verified ? 'text-green-600' : 'text-red-600'}`}>
                {ode.verified ? (
                  <><Check className="w-3 h-3 mr-1" /> Verified</>
                ) : (
                  <><X className="w-3 h-3 mr-1" /> Not Verified</>
                )}
              </span>
            )}
          </div>

          <div className="mb-2">
            <LatexRenderer latex={ode.ode} />
          </div>

          {expanded && (
            <div className="mt-4 space-y-3 border-t pt-3">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-600">Solution:</span>
                  <button
                    onClick={() => copyToClipboard(ode.solution)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                </div>
                <LatexRenderer latex={ode.solution} />
              </div>

              {ode.parameters && (
                <div>
                  <span className="text-sm font-medium text-gray-600">Parameters:</span>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {Object.entries(ode.parameters).map(([key, value]) => (
                      <span key={key} className="px-2 py-1 bg-gray-100 text-xs rounded">
                        {key} = {value}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {ode.verification && (
                <div>
                  <span className="text-sm font-medium text-gray-600">Verification:</span>
                  <div className="mt-1 text-sm text-gray-700">
                    Method: {ode.verification.method} | 
                    Confidence: {(ode.verification.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {ode.complexity !== undefined && (
                <div>
                  <span className="text-sm font-medium text-gray-600">
                    Complexity Score: 
                  </span>
                  <span className="ml-2 text-sm">{ode.complexity}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="ml-4 text-gray-400 hover:text-gray-600"
        >
          {expanded ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  )
}
