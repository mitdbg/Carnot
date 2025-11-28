import { useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { useAuth0 } from '@auth0/auth0-react'
import Layout from './components/Layout'
import DataManagementPage from './pages/DataManagementPage'
import DatasetCreatorPage from './pages/DatasetCreatorPage'
import UserChatPage from './pages/UserChatPage'

// read the organization ID from environment variables
const ORGANIZATION_ID = import.meta.env.VITE_AUTH0_ORGANIZATION_ID;

function App() {
  const { isAuthenticated, isLoading, error, loginWithRedirect } = useAuth0()

  // automatically redirect unauthenticated users to Auth0
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      // prepare the organization parameter
      const loginOptions = {};
      if (ORGANIZATION_ID) {
        loginOptions.organization = ORGANIZATION_ID;
        console.log(`Initiating login for organization: ${ORGANIZATION_ID}`);
      } else {
        // log a warning if the organization ID is not set
        console.warn('AUTH0_ORGANIZATION_ID environment variable is not set. Login may not be properly scoped.');
      }

      // perform the redirect with the configured options
      loginWithRedirect(loginOptions);
    }
  }, [isLoading, isAuthenticated, loginWithRedirect])

  if (isLoading) return <div className="p-8 text-center text-lg text-gray-600">Loading…</div>
  if (error) return <div className="p-8 text-center text-lg text-red-600">Error: {error.message}</div>

  // while redirecting, or if not authenticated yet, render a tiny message
  if (!isAuthenticated) {
    return <div className="p-8 text-center text-lg text-indigo-600">Redirecting to login...</div>
  }

  // authenticated layout
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

export default App;
