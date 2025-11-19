import { useState, useEffect } from 'react'
import { ChevronRight, ChevronDown, Folder, File, Loader2, Home, CheckSquare, Square } from 'lucide-react'
import { filesApi } from '../../services/api'

function FileBrowser({ selectedFiles, onFileToggle }) {
  const [currentPath, setCurrentPath] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedDirs, setExpandedDirs] = useState(new Set())
  const [directorySelectionState, setDirectorySelectionState] = useState(new Map()) // path -> boolean

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

  const handleCheckboxChange = (item, event) => {
    event.stopPropagation() // Prevent navigation when clicking checkbox
    
    if (item.is_directory) {
      handleDirectoryToggle(item)
    } else {
      onFileToggle(item.path, item.name)
    }
  }

  const isFileSelected = (item) => {
    if (item.is_directory) return false
    const key = `${item.path}||${item.name}`
    return selectedFiles.has(key)
  }

  const checkDirectorySelection = async (directoryPath) => {
    // Check if all files in this directory (and subdirectories) are selected
    try {
      const allFiles = await getAllFilesInDirectory(directoryPath)
      if (allFiles.length === 0) return false
      const allSelected = allFiles.every(file => {
        const key = `${file.path}||${file.name}`
        return selectedFiles.has(key)
      })
      setDirectorySelectionState(prev => new Map(prev).set(directoryPath, allSelected))
      return allSelected
    } catch (err) {
      return false
    }
  }

  // Update directory selection states when selectedFiles or items change
  useEffect(() => {
    const updateDirectoryStates = async () => {
      const newState = new Map()
      for (const item of items) {
        if (item.is_directory) {
          try {
            const allFiles = await getAllFilesInDirectory(item.path)
            if (allFiles.length > 0) {
              const allSelected = allFiles.every(file => {
                const key = `${file.path}||${file.name}`
                return selectedFiles.has(key)
              })
              newState.set(item.path, allSelected)
            } else {
              newState.set(item.path, false)
            }
          } catch (err) {
            newState.set(item.path, false)
          }
        }
      }
      setDirectorySelectionState(newState)
    }
    if (items.length > 0) {
      updateDirectoryStates()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFiles, items])

  const getAllFilesInDirectory = async (dirPath) => {
    const allFiles = []
    
    const loadDirRecursive = async (path) => {
      try {
        const response = await filesApi.browse(path)
        const items = response.data.items
        
        for (const item of items) {
          if (item.is_directory) {
            // Recursively load subdirectory
            const subPath = path ? `${path}/${item.name}` : item.name
            await loadDirRecursive(subPath)
          } else {
            // Add file to list
            allFiles.push({
              path: item.path,
              name: item.name
            })
          }
        }
      } catch (err) {
        console.error(`Failed to load directory ${path}:`, err)
      }
    }
    
    await loadDirRecursive(dirPath)
    return allFiles
  }

  const handleDirectoryToggle = async (directory) => {
    const allFiles = await getAllFilesInDirectory(directory.path)
    const allSelected = allFiles.every(file => {
      const key = `${file.path}||${file.name}`
      return selectedFiles.has(key)
    })
    
    const newSelected = new Set(selectedFiles)
    
    if (allSelected) {
      // Deselect all files in directory
      allFiles.forEach(file => {
        const key = `${file.path}||${file.name}`
        newSelected.delete(key)
      })
    } else {
      // Select all files in directory
      allFiles.forEach(file => {
        const key = `${file.path}||${file.name}`
        newSelected.add(key)
      })
    }
    
    onFileToggle(null, null, newSelected)
  }

  const getCurrentDirectoryFiles = () => {
    return items.filter(item => !item.is_directory)
  }

  const getCurrentDirectoryFolders = () => {
    return items.filter(item => item.is_directory)
  }

  const areAllItemsSelected = async () => {
    const files = getCurrentDirectoryFiles()
    const folders = getCurrentDirectoryFolders()
    
    if (files.length === 0 && folders.length === 0) return false
    
    // Check all files are selected
    const allFilesSelected = files.length === 0 || files.every(file => isFileSelected(file))
    
    // Check all folders are fully selected
    let allFoldersSelected = true
    for (const folder of folders) {
      const folderSelected = await checkDirectorySelection(folder.path)
      if (!folderSelected) {
        allFoldersSelected = false
        break
      }
    }
    
    return allFilesSelected && allFoldersSelected
  }

  const handleSelectAll = async () => {
    const files = getCurrentDirectoryFiles()
    const folders = getCurrentDirectoryFolders()
    const allSelected = await areAllItemsSelected()
    
    const newSelected = new Set(selectedFiles)
    
    if (allSelected) {
      // Deselect all files in current directory
      files.forEach(file => {
        const key = `${file.path}||${file.name}`
        newSelected.delete(key)
      })
      
      // Deselect all folders (and their contents)
      for (const folder of folders) {
        const allFiles = await getAllFilesInDirectory(folder.path)
        allFiles.forEach(file => {
          const key = `${file.path}||${file.name}`
          newSelected.delete(key)
        })
      }
    } else {
      // Select all files in current directory
      files.forEach(file => {
        const key = `${file.path}||${file.name}`
        newSelected.add(key)
      })
      
      // Select all folders (and their contents)
      for (const folder of folders) {
        const allFiles = await getAllFilesInDirectory(folder.path)
        allFiles.forEach(file => {
          const key = `${file.path}||${file.name}`
          newSelected.add(key)
        })
      }
    }
    
    // Update all at once
    onFileToggle(null, null, newSelected)
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

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <Folder className="w-5 h-5" />
          File Browser
        </h2>
        {items.length > 0 && (
          <button
            onClick={handleSelectAll}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-primary-600 hover:bg-primary-50 rounded-lg transition-colors border border-primary-200"
          >
            {(() => {
              const files = getCurrentDirectoryFiles()
              const folders = getCurrentDirectoryFolders()
              const allFilesSelected = files.length === 0 || files.every(f => isFileSelected(f))
              const allFoldersSelected = folders.length === 0 || folders.every(f => directorySelectionState.get(f.path) === true)
              return allFilesSelected && allFoldersSelected
            })() ? (
              <>
                <CheckSquare className="w-4 h-4" />
                Deselect All
              </>
            ) : (
              <>
                <Square className="w-4 h-4" />
                Select All
              </>
            )}
          </button>
        )}
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
                {/* Checkbox (for both files and directories) */}
                <input
                  type="checkbox"
                  checked={item.is_directory 
                    ? (directorySelectionState.get(item.path) || false)
                    : isFileSelected(item)}
                  onChange={(e) => handleCheckboxChange(item, e)}
                  className="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
                />

                {/* Icon and Name */}
                <button
                  onClick={() => handleItemClick(item)}
                  className="flex items-center gap-2 flex-1 text-left"
                  disabled={!item.is_directory}
                  onMouseEnter={() => {
                    if (item.is_directory) {
                      checkDirectorySelection(item.path)
                    }
                  }}
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

