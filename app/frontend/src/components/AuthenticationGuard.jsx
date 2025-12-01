// components/AuthenticationGuard.jsx
import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { Navigate } from 'react-router-dom';

function AuthenticationGuard({ children }) {
  const { isAuthenticated, isLoading, loginWithRedirect } = useAuth0();
  
  // get organization ID from environment variable
  const ORGANIZATION_ID = process.env.AUTH0_ORGANIZATION_ID;
  const loginOptions = ORGANIZATION_ID ? { organization: ORGANIZATION_ID } : {};

  // display message when loading
  if (isLoading) {
    return <div className="p-8 text-center text-lg text-gray-600">Loading…</div>;
  }

  // not authenticated, initiate login
  if (!isAuthenticated) {
    // Initiate the Auth0 login flow directly.
    // Auth0 will redirect the user to the IdP, and upon success/failure, 
    // bring them back to the /login page (or whatever the redirect_uri resolves to).
    loginWithRedirect(loginOptions);
    
    // While the redirect is happening, display a message.
    return <div className="p-8 text-center text-lg text-indigo-600">Redirecting to login...</div>;
  }

  // authenticated, render children
  return children;
}

export default AuthenticationGuard;
