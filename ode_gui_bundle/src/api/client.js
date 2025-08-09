import axios from 'axios'
import toast from 'react-hot-toast'
import { API_BASE, API_KEY } from '../config'

const client = axios.create({
  baseURL: API_BASE || window.location.origin,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY && { 'X-API-Key': API_KEY }),
  },
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message
    toast.error(message)
    return Promise.reject(error)
  }
)

export const api = {
  // System
  getInfo: () => client.get('/api/info').then(r => r.data),
  getHealth: () => client.get('/health').then(r => r.data),
  getMetrics: () => client.get('/metrics').then(r => r.data),

  // Generators & Functions
  getGenerators: () => client.get('/api/generators').then(r => r.data),
  getFunctions: () => client.get('/api/functions').then(r => r.data),

  // Generation
  generate: (data) => client.post('/api/generate', data).then(r => r.data),
  batchGenerate: (data) => client.post('/api/batch_generate', data).then(r => r.data),

  // Verification
  verify: (data) => client.post('/api/verify', data).then(r => r.data),

  // Datasets
  listDatasets: () => client.get('/api/datasets').then(r => r.data),
  createDataset: (odes, name) => 
    client.post('/api/datasets/create', odes, { params: { name } }).then(r => r.data),
  downloadDataset: (name, format = 'jsonl') => 
    client.get(`/api/datasets/${name}/download`, { 
      params: { format },
      responseType: 'blob' 
    }),

  // Jobs
  getJob: (jobId) => client.get(`/api/jobs/${jobId}`).then(r => r.data),
  listJobs: (status, limit = 100) => 
    client.get('/api/jobs', { params: { status, limit } }).then(r => r.data),
  cancelJob: (jobId) => client.delete(`/api/jobs/${jobId}`).then(r => r.data),

  // ML (if available)
  trainModel: (data) => client.post('/api/ml/train', data).then(r => r.data),
  generateWithML: (data) => client.post('/api/ml/generate', data).then(r => r.data),
  listModels: () => client.get('/api/ml/models').then(r => r.data),
}
