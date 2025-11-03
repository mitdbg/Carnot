import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import DataManagementPage from './pages/DataManagementPage'
import DatasetCreatorPage from './pages/DatasetCreatorPage'
import UserChatPage from './pages/UserChatPage'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<DataManagementPage />} />
          <Route path="/datasets/create" element={<DatasetCreatorPage />} />
          <Route path="/chat" element={<UserChatPage />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

