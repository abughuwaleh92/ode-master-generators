import React, { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { api } from '../../api/client'
import { Play, Download, Settings, RefreshCw } from 'lucide-react'
import toast from 'react-hot-toast'
import ODEDisplay from '../common/ODEDisplay'
import LoadingSpinner from '../common/LoadingSpinner'

export default function GeneratorPanel() {
  const [config, setConfig] = useState({
    generator: '',
    function: '',
    parameters: {
      alpha: 1.0,
      beta: 1.0,
      M: 0.0,
      q: 2,
      v: 3,
      a: 2,
    },
    count: 1,
    verify: true,
  })

  const [results, setResults] = useState(null)
  const [jobId, setJobId] = useState(null)

  const { data: generators, isLoading: loadingGenerators } = useQuery(
    'generators',
    api.getGenerators
  )

  const { data: functions, isLoading: loadingFunctions } = useQuery(
    'functions',
    api.getFunctions
  )

  const generateMutation = useMutation(api.generate, {
    onSuccess: (data) => {
      setJobId(data.job_id)
      toast.success('Generation started!')
      pollJob(data.job_id)
    },
  })

  const pollJob = async (id) => {
    const interval = setInterval(async () => {
      try {
        const job = await api.getJob(id)
        
        if (job.status === 'completed') {
          clearInterval(interval)
          setResults(job.results)
          toast.success('Generation completed!')
        } else if (job.status === 'failed') {
          clearInterval(interval)
          toast.error(`Generation failed: ${job.error}`)
        }
      } catch (error) {
        clearInterval(interval)
      }
    }, 1000)
  }

  const handleGenerate = () => {
    if (!config.generator || !config.function) {
      toast.error('Please select generator and function')
      return
    }
    generateMutation.mutate(config)
  }

  const handleDownload = () => {
    if (!results) return

    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `odes_${Date.now()}.json`
    a.click()
  }

  if (loadingGenerators || loadingFunctions) {
    return <LoadingSpinner />
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-6">ODE Generator</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Configuration */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Generator
              </label>
              <select
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={config.generator}
                onChange={(e) => setConfig({ ...config, generator: e.target.value })}
              >
                <option value="">Select generator...</option>
                {generators?.all?.map((gen) => (
                  <option key={gen} value={gen}>
                    {gen}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Function
              </label>
              <select
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={config.function}
                onChange={(e) => setConfig({ ...config, function: e.target.value })}
              >
                <option value="">Select function...</option>
                {functions?.functions?.map((func) => (
                  <option key={func} value={func}>
                    {func}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Count
              </label>
              <input
                type="number"
                min="1"
                max="100"
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                value={config.count}
                onChange={(e) => setConfig({ ...config, count: parseInt(e.target.value) })}
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="verify"
                className="mr-2"
                checked={config.verify}
                onChange={(e) => setConfig({ ...config, verify: e.target.checked })}
              />
              <label htmlFor="verify" className="text-sm">
                Verify solutions
              </label>
            </div>
          </div>

          {/* Parameters */}
          <div className="space-y-3">
            <h3 className="font-semibold flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              Parameters
            </h3>
            
            {Object.entries(config.parameters).map(([key, value]) => (
              <div key={key} className="flex items-center">
                <label className="w-16 text-sm">{key}:</label>
                <input
                  type="number"
                  step="0.1"
                  className="flex-1 px-2 py-1 border rounded"
                  value={value}
                  onChange={(e) => setConfig({
                    ...config,
                    parameters: {
                      ...config.parameters,
                      [key]: parseFloat(e.target.value),
                    },
                  })}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={handleGenerate}
            disabled={generateMutation.isLoading}
            className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center"
          >
            {generateMutation.isLoading ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Generate
              </>
            )}
          </button>

          {results && (
            <button
              onClick={handleDownload}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center"
            >
              <Download className="w-5 h-5 mr-2" />
              Download
            </button>
          )}
        </div>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold mb-4">
            Generated ODEs ({results.length})
          </h3>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {results.map((ode, idx) => (
              <ODEDisplay key={idx} ode={ode} index={idx + 1} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
