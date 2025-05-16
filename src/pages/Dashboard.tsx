import React from 'react';
import DataGrid from '../components/DataGrid';
import PlayerPerformanceChart from '../components/Chart'; // Import the new Chart component

const Dashboard = () => {
  return (
    <div>
      <h1>WNBA Player Dashboard</h1>
      <p>Displaying player stats and recommendations.</p>
      
      <div style={{ marginBottom: '20px' }}>
        <DataGrid />
      </div>
      
      <div>
        <h2>Performance Charts</h2>
        {/* TODO: Add controls to select player/stat for the chart */}
        <PlayerPerformanceChart playerName="Sample Player" />
      </div>
    </div>
  );
};

export default Dashboard; 