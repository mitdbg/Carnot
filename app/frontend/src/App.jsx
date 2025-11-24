import { useEffect, useState } from 'react';
import { Routes, Route, useSearchParams } from 'react-router-dom';
import { useAuth0 } from '@auth0/auth0-react';
import Layout from './components/Layout';
import DataManagementPage from './pages/DataManagementPage';
import DatasetCreatorPage from './pages/DatasetCreatorPage';
import UserChatPage from './pages/UserChatPage';
import ErrorPopup from './components/ErrorPopup';

// read the organization ID from the environment variable
const ORGANIZATION_ID = process.env.AUTH0_ORGANIZATION_ID;
const INVALID_DOMAIN_ERROR = "InvalidEmailDomain";

function App() {
  const { isAuthenticated, isLoading, error, loginWithRedirect } = useAuth0();
  const [searchParams] = useSearchParams();
  const [showErrorPopup, setShowErrorPopup] = useState(false);

  // get the error details from the URL if they exist
  const urlError = searchParams.get("error");
  const urlErrorDescription = searchParams.get("error_description");

  // useEffect to handle Auth0 redirect errors and show the custom message
  useEffect(() => {
    // handle the case where a user logs in with an invalid email domain
    if (urlError === "access_denied" && urlErrorDescription === INVALID_DOMAIN_ERROR) {
      // clear the query parameters from the URL to avoid continuous pop-up/redirect issues
      window.history.replaceState({}, document.title, window.location.pathname);

      // set state to show the error pop-up
      setShowErrorPopup(true);

      return;
    }

    // handle the standard unauthenticated redirect
    if (!isLoading && !isAuthenticated && !urlError) {
      const loginOptions = {};
      if (ORGANIZATION_ID) {
        loginOptions.organization = ORGANIZATION_ID;
        console.log(`Initiating login for organization: ${ORGANIZATION_ID}`);
      } else {
        console.warn('REACT_APP_AUTH0_ORGANIZATION_ID environment variable is not set. Login may not be properly scoped.');
      }

      loginWithRedirect(loginOptions);
    }
  }, [isLoading, isAuthenticated, loginWithRedirect, urlError, urlErrorDescription]);


  if (isLoading) return <div className="p-8 text-center text-lg text-gray-600">Loading…</div>
  // handle general Auth0 SDK errors (not the custom domain denial)
  if (error) return <div className="p-8 text-center text-lg text-red-600">Error: {error.message}</div>

  // if the specific domain denial error is present in the URL, render the error popup
  if (showErrorPopup) {
    // render the pop-up and return, preventing the app content or further redirects
    return (
      <ErrorPopup
        message="Your email domain is invalid."
        onClose={() => setShowErrorPopup(false)}
      />
    );
  }

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
