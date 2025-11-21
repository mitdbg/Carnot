import { useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import Layout from './components/Layout'
import DataManagementPage from './pages/DataManagementPage'
import DatasetCreatorPage from './pages/DatasetCreatorPage'
import UserChatPage from './pages/UserChatPage'

function App() {
  const { isAuthenticated, isLoading, error, loginWithRedirect } = useAuth0()

  // Automatically redirect unauthenticated users to Auth0
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      loginWithRedirect()
    }
  }, [isLoading, isAuthenticated, loginWithRedirect])

  if (isLoading) return <div>Loading…</div>
  if (error) return <div>Error: {error.message}</div>

  // While redirecting, or if not authenticated yet, render nothing or a tiny message
  if (!isAuthenticated) {
    return <div>Redirecting to login…</div>
  }

  // Authenticated layout
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DataManagementPage />} />
        <Route path="/datasets/create" element={<DatasetCreatorPage />} />
        <Route path="/chat" element={<UserChatPage />} />
      </Routes>
    </Layout>
  )
}

export default App
