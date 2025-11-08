import { useState, useEffect } from 'react'
import { ChevronRight, ChevronDown, Folder, File, Loader2, Home } from 'lucide-react'
import { filesApi } from '../../services/api'

function FileBrowser({ selectedFiles, onFileToggle }) {
  const [currentPath, setCurrentPath] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedDirs, setExpandedDirs] = useState(new Set())

  useEffect(() => {
    loadDirectory(currentPath)
  }, [currentPath])

  const loadDirectory = async (path) => {
    try {
      setLoading(true)
      setError(null)
      const response = await filesApi.browse(path)
      setItems(response.data.items)
    } catch (err) {
      setError('Failed to load directory: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleItemClick = (item) => {
    if (item.is_directory) {
      setCurrentPath(item.path)
      setExpandedDirs(new Set([...expandedDirs, item.path]))
    }
  }

  const makeSelectionKey = (item) => `${item.path}||${item.name}||${item.is_directory ? 'dir' : 'file'}`

  const handleCheckboxChange = (item) => {
    onFileToggle({
      path: item.path,
      name: item.name,
      is_directory: item.is_directory,
    })
  }

  const isFileSelected = (item) => {
    const key = makeSelectionKey(item)
    return selectedFiles.has(key)
  }

  const navigateUp = () => {
    if (currentPath) {
      const parts = currentPath.split('/')
      parts.pop()
      setCurrentPath(parts.join('/'))
    }
  }

  const navigateToRoot = () => {
    setCurrentPath('')
  }

  const getBreadcrumbs = () => {
    if (!currentPath) return []
    return currentPath.split('/').filter(Boolean)
  }

  const handleSelectAll = () => {
    items.forEach((item) => {
      if (item.name === '..') {
        return
      }
      if (!isFileSelected(item)) {
        handleCheckboxChange(item)
      }
    })
  }

  const handleClearSelection = () => {
    items.forEach((item) => {
      if (item.name === '..') {
        return
      }
      if (isFileSelected(item)) {
        handleCheckboxChange(item)
      }
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <Folder className="w-5 h-5" />
          File Browser
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={handleSelectAll}
            className="text-sm px-3 py-1 rounded border border-gray-300 hover:bg-gray-100 transition-colors"
            type="button"
          >
            Select All
          </button>
          <button
            onClick={handleClearSelection}
            className="text-sm px-3 py-1 rounded border border-gray-300 hover:bg-gray-100 transition-colors"
            type="button"
          >
            Clear Selection
          </button>
        </div>
      </div>

      {/* Breadcrumb Navigation */}
      <div className="mb-4 flex items-center gap-2 text-sm">
        <button
          onClick={navigateToRoot}
          className="flex items-center gap-1 px-2 py-1 hover:bg-gray-100 rounded transition-colors"
        >
          <Home className="w-4 h-4" />
          <span>Root</span>
        </button>
        {getBreadcrumbs().map((crumb, index) => {
          const crumbPath = getBreadcrumbs().slice(0, index + 1).join('/')
          return (
            <div key={index} className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-gray-400" />
              <button
                onClick={() => setCurrentPath(crumbPath)}
                className="px-2 py-1 hover:bg-gray-100 rounded transition-colors"
              >
                {crumb}
              </button>
            </div>
          )
        })}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-600 px-3 py-2 rounded mb-4 text-sm">
          {error}
        </div>
      )}

      {/* File List */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Folder className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>No items in this directory</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
            {/* Back Button */}
            {currentPath && (
              <button
                onClick={navigateUp}
                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors text-left"
              >
                <ChevronDown className="w-5 h-5 text-gray-400 transform rotate-90" />
                <span className="text-gray-600">..</span>
              </button>
            )}

            {/* Items */}
            {items.map((item, index) => (
              <div
                key={index}
                className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors"
              >
                {/* Checkbox for files and directories */}
                <input
                  type="checkbox"
                  checked={isFileSelected(item)}
                  onChange={() => handleCheckboxChange(item)}
                  className="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
                />

                {/* Icon and Name */}
                <button
                  onClick={() => handleItemClick(item)}
                  className="flex items-center gap-2 flex-1 text-left"
                  disabled={!item.is_directory}
                >
                  {item.is_directory ? (
                    <>
                      <ChevronRight className="w-5 h-5 text-gray-400" />
                      <Folder className="w-5 h-5 text-primary-500" />
                    </>
                  ) : (
                    <File className="w-5 h-5 text-gray-400 ml-5" />
                  )}
                  <span className={`${item.is_directory ? 'font-medium text-gray-800' : 'text-gray-600'}`}>
                    {item.name}
                  </span>
                </button>

                {/* File Size */}
                {!item.is_directory && item.size !== null && (
                  <span className="text-xs text-gray-400">
                    {formatFileSize(item.size)}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

export default FileBrowser

