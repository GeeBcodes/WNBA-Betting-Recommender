import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css'; // This will point to src/index.css
import App from './App.tsx'; // This will point to src/App.tsx

// AG Grid imports
// import 'ag-grid-community/styles/ag-grid.css'; // Core grid CSS - REMOVED as per v33 theming guidelines
import 'ag-grid-community/styles/ag-theme-quartz.css'; // Modern Theme
import { ModuleRegistry, AllCommunityModule } from 'ag-grid-community';

// Register AG Grid modules
ModuleRegistry.registerModules([AllCommunityModule]);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
); 