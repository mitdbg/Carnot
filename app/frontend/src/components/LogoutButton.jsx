import { useAuth0 } from '@auth0/auth0-react';

const LogoutButton = () => {
  const { logout } = useAuth0();

  const handleLogout = () => {
    // clear any application-specific storage
    localStorage.clear();
    sessionStorage.clear();

    // Auth0 logout
    logout({
      logoutParams: {
        returnTo: window.location.origin,
      },
    });
  };

  return <button onClick={handleLogout}>Log Out</button>;
};

export default LogoutButton;
