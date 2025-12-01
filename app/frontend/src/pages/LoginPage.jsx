// pages/LoginPage.jsx
import React, { useEffect, useState, useMemo } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useSearchParams } from 'react-router-dom';
import ErrorPopup from '../components/ErrorPopup';

const ORGANIZATION_ID = process.env.AUTH0_ORGANIZATION_ID;
const INVALID_DOMAIN_ERROR = "InvalidEmailDomain";

const LoginPage = () => {
    const { loginWithRedirect } = useAuth0();
    const [searchParams] = useSearchParams();
    const [showErrorPopup, setShowErrorPopup] = useState(false);

    const urlError = searchParams.get("error");
    const urlErrorDescription = searchParams.get("error_description");

    const loginOptions = useMemo(() => {
        const options = {};
        if (ORGANIZATION_ID) {
            options.organization = ORGANIZATION_ID;
        }
        return options;
    }, []);
    
    // Check for the denial error when this page loads
    useEffect(() => {
        if (urlError === "access_denied" && urlErrorDescription === INVALID_DOMAIN_ERROR) {
            // Clear the query parameters from the URL
            window.history.replaceState({}, document.title, window.location.pathname);
            setShowErrorPopup(true);
        }
    }, [urlError, urlErrorDescription]);

    if (showErrorPopup) {
        return (
            <ErrorPopup 
                message="Your email domain is invalid." 
                onClose={() => setShowErrorPopup(false)}
            />
        );
    }

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
            <div className="text-center p-8 bg-white shadow-xl rounded-lg">
                <h1 className="text-3xl font-bold text-indigo-600 mb-4">Welcome</h1>
                <p className="text-gray-600 mb-6">
                    Log in.
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

export default LoginPage;
