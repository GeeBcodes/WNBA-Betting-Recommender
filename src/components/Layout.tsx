import React from 'react';
import { Outlet } from 'react-router-dom';

interface LayoutProps {
  // children prop is implicitly handled by Outlet for nested routes
}

const Layout: React.FC<LayoutProps> = () => {
  return (
    <div className="app-layout">
      {/* <header>App Header / Navbar could go here</header> */}
      <main style={{ padding: '20px' }}>
        <Outlet /> {/* Child routes will render here */}
      </main>
      {/* <footer>App Footer could go here</footer> */}
    </div>
  );
};

export default Layout; 