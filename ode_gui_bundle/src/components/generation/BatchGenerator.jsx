import React, { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { api } from '../../api/client'
import { Play, Download, Save, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import LoadingSpinner from '../common/LoadingSpinner'

export default function BatchGenerator() {
  const [config, setConfig] = useState({
    generators: [],
    functions: [],
    samples_per_combination: 5,
    parameter_ranges: {
      alpha: [0, 0.5, 1, 1.5, 2],
      beta: [0.5, 1, 1.5, 2],
      M: [0, 0.5, 1],
      q: [2, 3],
      v: [2, 3, 4],
      a: [2, 3, 4],
    },
    verify: true,
    save_dataset: true,
    dataset_name: '',
  })

  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)

  const { data: generators } = useQuery('generators', api.getGenerators)
  const { data: functions } = useQuery('functions', api.getFunctions)

  const batchMutation = useMutation(api.batchGenerate, {
    onSuccess: (data) => {
      setJobId(data.job_id)
      toast.success(`Batch generation started! Expected ${data.total_expected} ODEs`)
      pollJob(data.job_id)
    },
  })

  const pollJob = async (id) => {
    const interval = setInterval(async () => {
      try {
        const job = await api.getJob(id)
        setJobStatus(job)
        
        if (job.status === 'completed') {
          clearInterval(interval)
          toast.success('Batch generation completed!')
        } else if (job.status === 'failed') {
          clearInterval(interval)
          toast.error(`Generation failed: ${job.error}`)
        }
      } catch (error) {
        clearInterval(interval)
      }
    }, 2000)
  }

  const handleGenerate = () => {
    if (config.generators.length === 0 || config.functions.length === 0) {
      toast.error('Please select at least one generator and function')
      return
    }
    batchMutation.mutate(config)
  }

  const toggleGenerator = (gen) => {
    setConfig(prev => ({
      ...prev,
      generators: prev.generators.includes(gen)
        ? prev.generators.filter(g => g !== gen)
        : [...prev.generators, gen]
    }))
  }

  const toggleFunction = (func) => {
    setConfig(prev => ({
      ...prev,
      functions: prev.functions.includes(func)
        ? prev.functions.filter(f => f !== func)
        : [...prev.functions, func]
    }))
  }

  const selectAllGenerators = () => {
    setConfig(prev => ({
      ...prev,
      generators: generators?.all || []
    }))
  }

  const selectAllFunctions = () => {
    setConfig(prev => ({
      ...prev,
      functions: functions?.functions || []
    }))
  }

  const expectedTotal = config.generators.length * config.functions.length * config.samples_per_combination

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-6">Batch ODE Generation</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Generators Selection */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold">Generators ({config.generators.length} selected)</h3>
              <button
                onClick={selectAllGenerators}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Select All
              </button>
            </div>
            
            <div className="border rounded-lg p-4 max-h-64 overflow-y-auto">
              <div className="space-y-2">
                <div className="font-medium text-sm text-gray-600 mb-2">Linear</div>
                {generators?.linear?.map(gen => (
                  <label key={gen} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.generators.includes(gen)}
                      onChange={() => toggleGenerator(gen)}
                      className="mr-2"
                    />
                    <span className="text-sm">{gen}</span>
                  </label>
                ))}
                
                <div className="font-medium text-sm text-gray-600 mt-3 mb-2">Nonlinear</div>
                {generators?.nonlinear?.map(gen => (
                  <label key={gen} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.generators.includes(gen)}
                      onChange={() => toggleGenerator(gen)}
                      className="mr-2"
                    />
                    <span className="text-sm">{gen}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Functions Selection */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold">Functions ({config.functions.length} selected)</h3>
              <button
                onClick={selectAllFunctions}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Select All
              </button>
            </div>
            
            <div className="border rounded-lg p-4 max-h-64 overflow-y-auto">
              <div className="grid grid-cols-2 gap-2">
                {functions?.functions?.map(func => (
                  <label key={func} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.functions.includes(func)}
                      onChange={(e) => toggleFunction(func)}
                      className="mr-2"
                    />
                    <span className="text-sm">{func}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Configuration */}
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Samples per Combination
              </label>
              <input
                type="number"
                min="1"
                max="50"
                className="input"
                value={config.samples_per_combination}
                onChange={(e) => setConfig({
                  ...config,
                  samples_per_combination: parseInt(e.target.value)
                })}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Dataset Name (optional)
              </label>
              <input
                type="text"
                className="input"
                placeholder="Auto-generated if empty"
                value={config.dataset_name}
                onChange={(e) => setConfig({
                  ...config,
                  dataset_name: e.target.value
                })}
              />
            </div>

            <div className="flex items-end space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.verify}
                  onChange={(e) => setConfig({ ...config, verify: e.target.checked })}
                  className="mr-2"
                />
                <span className="text-sm">Verify</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.save_dataset}
                  onChange={(e) => setConfig({ ...config, save_dataset: e.target.checked })}
                  className="mr-2"
                />
                <span className="text-sm">Save Dataset</span>
              </label>
            </div>
          </div>

          {/* Parameter Ranges */}
          <div>
            <h3 className="font-semibold mb-3">Parameter Ranges</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(config.parameter_ranges).map(([key, values]) => (
                <div key={key}>
                  <label className="block text-sm font-medium mb-1">{key}</label>
                  <input
                    type="text"
                    className="input text-sm"
                    value={values.join(', ')}
                    onChange={(e) => {
                      const newValues = e.target.value
                        .split(',')
                        .map(v => parseFloat(v.trim()))
                        .filter(v => !isNaN(v))
                      
                      setConfig({
                        ...config,
                        parameter_ranges: {
                          ...config.parameter_ranges,
                          [key]: newValues
                        }
                      })
                    }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Expected Output */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 mr-2" />
              <div>
                <p className="text-sm font-medium text-blue-900">
                  Expected Output
                </p>
                <p className="text-sm text-blue-700 mt-1">
                  {expectedTotal} ODEs will be generated
                  ({config.generators.length} generators × {config.functions.length} functions × {config.samples_per_combination} samples)
                </p>
              </div>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={batchMutation.isLoading || expectedTotal === 0}
            className="w-full btn-primary"
          >
            {batchMutation.isLoading ? (
              <span className="flex items-center justify-center">
                <LoadingSpinner size="sm" className="mr-2" />
                Generating...
              </span>
            ) : (
              <span className="flex items-center justify-center">
                <Play className="w-5 h-5 mr-2" />
                Generate {expectedTotal} ODEs
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Job Status */}
      {jobStatus && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Generation Progress</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Status</span>
              <span className={`px-2 py-1 rounded text-xs ${
                jobStatus.status === 'completed' ? 'bg-green-100 text-green-800' :
                jobStatus.status === 'failed' ? 'bg-red-100 text-red-800' :
                jobStatus.status === 'running' ? 'bg-blue-100 text-blue-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {jobStatus.status}
              </span>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Progress</span>
                <span>{jobStatus.progress?.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${jobStatus.progress}%` }}
                />
              </div>
            </div>

            {jobStatus.metadata && (
              <div className="text-sm text-gray-600">
                <p>Current: {jobStatus.metadata.current} / {jobStatus.metadata.total}</p>
                {jobStatus.metadata.generator && (
                  <p>Processing: {jobStatus.metadata.generator} - {jobStatus.metadata.function}</p>
                )}
              </div>
            )}

            {jobStatus.status === 'completed' && jobStatus.results && (
              <div className="bg-green-50 rounded-lg p-4">
                <h4 className="font-medium text-green-900 mb-2">Results</h4>
                <div className="space-y-1 text-sm text-green-700">
                  <p>Total Generated: {jobStatus.results.total_generated}</p>
                  <p>Verified: {jobStatus.results.summary?.verified || 0}</p>
                  {jobStatus.results.dataset_name && (
                    <p>Dataset: {jobStatus.results.dataset_name}</p>
                  )}
                </div>
                
                {jobStatus.results.dataset_name && (
                  <button
                    onClick={() => {
                      api.downloadDataset(jobStatus.results.dataset_name)
                        .then(response => {
                          const url = window.URL.createObjectURL(response.data)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = `${jobStatus.results.dataset_name}.jsonl`
                          a.click()
                        })
                    }}
                    className="mt-3 flex items-center text-green-700 hover:text-green-900"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download Dataset
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
