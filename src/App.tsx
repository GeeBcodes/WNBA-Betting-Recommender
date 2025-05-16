import React from 'react';
import { RouterProvider } from "react-router-dom";
import { router } from "./routes";
// import './App.css'; // Removed as App.css was part of the old nested structure and likely not needed now

function App() {
  return <RouterProvider router={router} />;
}

export default App; 