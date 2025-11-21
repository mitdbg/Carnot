import { Routes, Route } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import Layout from './components/Layout'
import DataManagementPage from './pages/DataManagementPage'
import DatasetCreatorPage from './pages/DatasetCreatorPage'
import UserChatPage from './pages/UserChatPage'
import LoginButton from './LoginButton'

function App() {
  const { isAuthenticated, isLoading, error } = useAuth0()

  if (isLoading) return <div>Loading…</div>
  if (error) return <div>Error: {error.message}</div>

  if (!isAuthenticated) {
    return (
      <div className="login-container">
        <h1>Welcome to Carnot</h1>
        <LoginButton />
      </div>
    )
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
