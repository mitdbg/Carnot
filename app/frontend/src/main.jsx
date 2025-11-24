import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './index.css'
import { Auth0Provider } from '@auth0/auth0-react';

const SUBDOMAIN = process.env.SUBDOMAIN;

const onRedirectCallback = () => {
  const params = new URLSearchParams(window.location.search);

  const err = params.get("error");
  const desc = params.get("error_description");

  // Detect the domain restriction denial
  if (err === "access_denied" && desc === "InvalidEmailDomain") {
    // Redirect the user to your landing page (logged-out state)
    window.location.replace(`https://auth.${SUBDOMAIN}.carnot-research.org`);
    return;
  }

  // Clear the error params from the URL (React SPA cleanup)
  window.history.replaceState({}, document.title, window.location.pathname);
};

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Auth0Provider
      domain={import.meta.env.VITE_AUTH0_DOMAIN}
      clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}
      authorizationParams={{
        redirect_uri: window.location.origin
      }}
      cacheLocation="localstorage"
      useRefreshTokens={true}
      onRedirectCallback={onRedirectCallback}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </Auth0Provider>
  </React.StrictMode>,
)
