import React from 'react';
import { createBrowserRouter } from "react-router-dom";
import Layout from "../components/Layout";
import Dashboard from "../pages/Dashboard";
import PlayerDetail from "../pages/PlayerDetail";
import OddsOverview from "../pages/OddsOverview";
import ParlayGenerator from "../pages/ParlayGenerator";

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Dashboard /> },
      { path: "/players/:playerId", element: <PlayerDetail /> },
      { path: "/odds", element: <OddsOverview /> },
      { path: "/parlay", element: <ParlayGenerator /> },
    ]
  }
]); 