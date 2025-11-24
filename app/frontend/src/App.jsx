// App.jsx (Final Structure)
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import DataManagementPage from './pages/DataManagementPage';
import DatasetCreatorPage from './pages/DatasetCreatorPage';
import UserChatPage from './pages/UserChatPage';
import LoginPage from './pages/LoginPage';
import AuthenticationGuard from './components/AuthenticationGuard';

function App() {
  return (
    <Layout>
      <Routes>
        {/* 1. Public Route: Landing and Error Handling */}
        <Route path="/" element={<LoginPage />} />
        
        {/* 2. Protected Routes: Use the Guard component */}
        <Route 
          path="/data" 
          element={
            <AuthenticationGuard component={<DataManagementPage />} />
          } 
        />
        <Route 
          path="/datasets/create" 
          element={
            <AuthenticationGuard component={<DatasetCreatorPage />} />
          } 
        />
        <Route 
          path="/chat" 
          element={
            <AuthenticationGuard component={<UserChatPage />} />
          } 
        />
        
        {/* Optional: Redirect any unknown path to the login page */}
        <Route path="*" element={<LoginPage />} />
      </Routes>
    </Layout>
  )
}

export default App;