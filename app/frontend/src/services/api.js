import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Files API
export const filesApi = {
  browse: (path = '') => api.get('/files/browse', { params: { path } }),
  upload: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  listUploaded: () => api.get('/files/uploaded'),
}

// Datasets API
export const datasetsApi = {
  list: () => api.get('/datasets'),
  create: (data) => api.post('/datasets', data),
  get: (id) => api.get(`/datasets/${id}`),
  update: (id, data) => api.put(`/datasets/${id}`, data),
  delete: (id) => api.delete(`/datasets/${id}`),
}

// Search API
export const searchApi = {
  search: (query, path = null) => api.post('/search', { query, path }),
}

export default api
