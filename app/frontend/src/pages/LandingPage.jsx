// pages/LandingPage.jsx
import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LandingPage = ({ loginOptions }) => {
  const { loginWithRedirect } = useAuth0();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <div className="text-center p-8 bg-white shadow-xl rounded-lg">
        <h1 className="text-3xl font-bold text-indigo-600 mb-4">Welcome to the App</h1>
        <p className="text-gray-600 mb-6">
          Please log in to continue. Only authorized email domains (@mit.edu, @gmail.com) are permitted.
        </p>
        <button
          onClick={() => loginWithRedirect(loginOptions)}
          className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition duration-150"
        >
          Log In
        </button>
      </div>
    </div>
  );
};

export default LandingPage;

