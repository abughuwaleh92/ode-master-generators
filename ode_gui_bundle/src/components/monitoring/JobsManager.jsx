import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { api } from '../../api/client'
import { 
  Briefcase, PlayCircle, CheckCircle, XCircle, 
  Clock, RefreshCw, Trash2, Filter 
} from 'lucide-react'
import { format } from 'date-fns'
import toast from 'react-hot-toast'

export default function JobsManager() {
  const [statusFilter, setStatusFilter] = useState(null)
  const queryClient = useQueryClient()

  const { data: jobs, isLoading } = useQuery(
    ['jobs', statusFilter],
    () => api.listJobs(statusFilter),
    { refetchInterval: 5000 }
  )

  const cancelMutation = useMutation(api.cancelJob, {
    onSuccess: () => {
      queryClient.invalidateQueries('jobs')
      toast.success('Job cancelled')
    },
  })

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'running':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />
      case 'queued':
        return <Clock className="w-5 h-5 text-yellow-500" />
      default:
        return <Briefcase className="w-5 h-5 text-gray-500" />
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'queued': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Jobs Manager</h2>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="w-5 h-5 text-gray-400" />
              <select
                className="select"
                value={statusFilter || ''}
                onChange={(e) => setStatusFilter(e.target.value || null)}
              >
                <option value="">All Status</option>
                <option value="queued">Queued</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </div>
          </div>
        </div>

        {/* Queue Statistics */}
        {jobs?.queue_stats && Object.keys(jobs.queue_stats).length > 0 && (
          <div className="mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            {Object.entries(jobs.queue_stats).map(([queue, count]) => (
              <div key={queue} className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold">{count}</div>
                <div className="text-sm text-gray-600 capitalize">{queue} Queue</div>
              </div>
            ))}
          </div>
        )}

        {/* Jobs List */}
        {isLoading ? (
          <div className="text-center py-8">Loading jobs...</div>
        ) : jobs?.jobs?.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Briefcase className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>No jobs found</p>
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.jobs.map((job) => (
              <div
                key={job.job_id}
                className="border rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {getStatusIcon(job.status)}
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-mono text-sm text-gray-600">
                          {job.job_id.slice(0, 8)}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>Created: {format(new Date(job.created_at), 'MMM d, HH:mm:ss')}</div>
                        
                        {job.status === 'running' && job.progress > 0 && (
                          <div>
                            <div className="flex justify-between text-xs mb-1">
                              <span>Progress</span>
                              <span>{job.progress.toFixed(1)}%</span>
                            </div>
                            <div className="w-48 bg-gray-200 rounded-full h-1.5">
                              <div
                                className="bg-blue-600 h-1.5 rounded-full"
                                style={{ width: `${job.progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                        
                        {job.eta && (
                          <div>ETA: {format(new Date(job.eta), 'HH:mm:ss')}</div>
                        )}
                        
                        {job.error && (
                          <div className="text-red-600">Error: {job.error}</div>
                        )}
                        
                        {job.metadata && Object.keys(job.metadata).length > 0 && (
                          <div className="text-xs mt-2">
                            {Object.entries(job.metadata).map(([key, value]) => (
                              <span key={key} className="inline-block mr-3">
                                {key}: {JSON.stringify(value)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {job.status === 'running' || job.status === 'queued' ? (
                    <button
                      onClick={() => cancelMutation.mutate(job.job_id)}
                      className="text-red-600 hover:text-red-800"
                      title="Cancel Job"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
