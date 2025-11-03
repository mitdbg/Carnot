import { useState, useEffect, useRef } from 'react'
import { Send, Database, CheckSquare, Square, AlertCircle, Loader2, XCircle, RotateCcw } from 'lucide-react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

function UserChatPage() {
  const [datasets, setDatasets] = useState([])
  const [selectedDatasets, setSelectedDatasets] = useState(new Set())
  const [messages, setMessages] = useState([])
  const [inputQuery, setInputQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const messagesEndRef = useRef(null)
  const abortControllerRef = useRef(null)
  
  // Generate session ID on component mount
  useEffect(() => {
    generateNewSession()
  }, [])

  useEffect(() => {
    loadDatasets()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  const generateNewSession = () => {
    const newSessionId = crypto.randomUUID()
    setSessionId(newSessionId)
    setMessages([])
    console.log('New session created:', newSessionId)
  }

  const loadDatasets = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/datasets/`)
      setDatasets(response.data)
    } catch (error) {
      console.error('Error loading datasets:', error)
    }
  }

  const toggleDataset = (datasetId) => {
    setSelectedDatasets(prev => {
      const newSet = new Set(prev)
      if (newSet.has(datasetId)) {
        newSet.delete(datasetId)
      } else {
        newSet.add(datasetId)
      }
      return newSet
    })
  }

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Query cancelled by user.'
      }])
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!inputQuery.trim()) {
      return
    }

    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Please select at least one dataset before submitting a query.'
      }])
      return
    }

    // Add user message
    const userMessage = {
      type: 'user',
      content: inputQuery
    }
    setMessages(prev => [...prev, userMessage])
    setInputQuery('')
    setIsLoading(true)

    try {
      // Create abort controller for this request
      abortControllerRef.current = new AbortController()

      const response = await fetch(`${API_BASE_URL}/api/query/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          dataset_ids: Array.from(selectedDatasets),
          session_id: sessionId
        }),
        signal: abortControllerRef.current.signal
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // Update session_id if received from server
              if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id)
                console.log('Session ID updated:', data.session_id)
              }
              
              if (data.type === 'status') {
                setMessages(prev => [...prev, {
                  type: 'status',
                  content: data.message
                }])
              } else if (data.type === 'result') {
                setMessages(prev => [...prev, {
                  type: 'result',
                  content: data.message,
                  csv_file: data.csv_file,
                  row_count: data.row_count
                }])
              } else if (data.type === 'error') {
                setMessages(prev => [...prev, {
                  type: 'error',
                  content: data.message
                }])
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request aborted')
      } else {
        console.error('Error executing query:', error)
        setMessages(prev => [...prev, {
          type: 'error',
          content: 'Failed to execute query. Please try again.'
        }])
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const renderMessage = (message, index) => {
    switch (message.type) {
      case 'user':
        return (
          <div key={index} className="flex justify-end mb-4">
            <div className="bg-primary-500 text-white rounded-lg px-4 py-2 max-w-[70%]">
              {message.content}
            </div>
          </div>
        )
      
      case 'assistant':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-gray-100 text-gray-800 rounded-lg px-4 py-2 max-w-[70%]">
              <div className="whitespace-pre-wrap">{message.content}</div>
              {message.files && message.files.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-300">
                  <p className="text-xs text-gray-600 mb-1">Files:</p>
                  {message.files.map((file, i) => (
                    <div key={i} className="text-xs text-gray-700 font-mono">
                      {file.filename}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )
      
      case 'status':
        return (
          <div key={index} className="flex justify-center mb-3">
            <div className="bg-blue-50 text-blue-700 rounded-full px-4 py-1 text-sm flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              {message.content}
            </div>
          </div>
        )
      
      case 'error':
        return (
          <div key={index} className="flex justify-center mb-4">
            <div className="bg-red-50 text-red-700 rounded-lg px-4 py-2 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {message.content}
            </div>
          </div>
        )
      
      case 'result':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-[80%]">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap mb-3 overflow-auto max-h-80">
                {message.content}
              </pre>
              {message.csv_file && (
                <a
                  href={`${API_BASE_URL}/api/query/download/${message.csv_file}`}
                  download={message.csv_file}
                  className="inline-flex items-center gap-2 bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Database className="w-4 h-4" />
                  Download Full CSV ({message.row_count} rows)
                </a>
              )}
            </div>
          </div>
        )
      
      default:
        return null
    }
  }

  return (
    <div className="h-[calc(100vh-4rem)] flex">
      {/* Left side - Chat */}
      <div className="flex-1 flex flex-col border-r border-gray-200">
        {/* Chat header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Query Chat</h1>
            <p className="text-sm text-gray-600 mt-1">
              Ask questions about your data {sessionId && <span className="text-xs text-gray-400">(Session: {sessionId.slice(0, 8)}...)</span>}
            </p>
          </div>
          <button
            onClick={generateNewSession}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:cursor-not-allowed text-gray-700 rounded-lg transition-colors"
            title="Start a new conversation"
          >
            <RotateCcw className="w-4 h-4" />
            New Conversation
          </button>
        </div>

        {/* Messages container */}
        <div className="flex-1 overflow-y-auto px-6 py-4 bg-gray-50">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Database className="w-16 h-16 text-gray-300 mb-4" />
              <h2 className="text-xl font-semibold text-gray-600 mb-2">
                No messages yet
              </h2>
              <p className="text-gray-500 max-w-md">
                Select datasets from the right panel and start asking questions about your data
              </p>
            </div>
          )}
          
          {messages.map((message, index) => renderMessage(message, index))}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask a question about your data..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              disabled={isLoading}
            />
            {isLoading ? (
              <button
                type="button"
                onClick={handleCancel}
                className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 flex items-center gap-2 transition-colors"
              >
                <XCircle className="w-4 h-4" />
                Cancel
              </button>
            ) : (
              <button
                type="submit"
                disabled={!inputQuery.trim() || selectedDatasets.size === 0}
                className="bg-primary-500 text-white px-6 py-2 rounded-lg hover:bg-primary-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            )}
          </form>
          
          {selectedDatasets.size === 0 && (
            <p className="text-xs text-amber-600 mt-2 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" />
              Select at least one dataset to enable queries
            </p>
          )}
        </div>
      </div>

      {/* Right side - Dataset selection */}
      <div className="w-80 bg-white flex flex-col">
        <div className="border-b border-gray-200 px-4 py-4">
          <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
            <Database className="w-5 h-5" />
            Datasets
          </h2>
          <p className="text-xs text-gray-600 mt-1">
            {selectedDatasets.size} of {datasets.length} selected
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          {datasets.length === 0 ? (
            <div className="text-center py-8">
              <Database className="w-12 h-12 text-gray-300 mx-auto mb-2" />
              <p className="text-sm text-gray-500">No datasets available</p>
              <p className="text-xs text-gray-400 mt-1">
                Create a dataset first
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {datasets.map(dataset => (
                <div
                  key={dataset.id}
                  onClick={() => toggleDataset(dataset.id)}
                  className={`
                    border rounded-lg p-3 cursor-pointer transition-all
                    ${selectedDatasets.has(dataset.id)
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                    }
                  `}
                >
                  <div className="flex items-start gap-2">
                    <div className="mt-0.5">
                      {selectedDatasets.has(dataset.id) ? (
                        <CheckSquare className="w-5 h-5 text-primary-500" />
                      ) : (
                        <Square className="w-5 h-5 text-gray-400" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-sm text-gray-800 truncate">
                        {dataset.name}
                      </h3>
                      {dataset.annotation && (
                        <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                          {dataset.annotation}
                        </p>
                      )}
                      <p className="text-xs text-gray-500 mt-1">
                        {dataset.file_count} file{dataset.file_count !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UserChatPage
