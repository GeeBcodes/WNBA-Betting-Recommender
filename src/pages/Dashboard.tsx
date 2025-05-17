import React, { useEffect, useState } from 'react';
import DataGrid from '../components/DataGrid';
import PlayerPerformanceChart from '../components/Chart'; // Import the new Chart component
import { getPredictions, postParlay, Prediction, ParlayData } from '../services/api'; // Import new functions and types

const Dashboard = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);
        const data = await getPredictions();
        setPredictions(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch predictions:", err);
        setError('Failed to load predictions. Please try again later.');
        setPredictions([]); // Clear predictions on error
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  const handleCreateParlay = async () => {
    // Placeholder for creating a parlay
    // In a real app, you would gather selected predictions from the DataGrid or other UI elements
    const sampleParlayData: ParlayData = {
      name: "My Sample Parlay",
      legs: [
        // Example: Add first two predictions to the parlay if available
        ...(predictions.length > 0 ? [{ prediction_id: predictions[0].id, type: 'over' as const }] : []),
        ...(predictions.length > 1 ? [{ prediction_id: predictions[1].id, type: 'under' as const }] : []),
      ],
    };

    if (sampleParlayData.legs.length === 0) {
      alert("Not enough predictions available to create a sample parlay.");
      return;
    }

    try {
      console.log("Attempting to post parlay:", sampleParlayData);
      const result = await postParlay(sampleParlayData);
      console.log('Parlay created successfully:', result);
      alert('Sample Parlay posted! Check the console for details.');
      // TODO: Optionally, refresh predictions or update UI based on result
    } catch (err) {
      console.error('Failed to post parlay:', err);
      alert('Failed to post sample parlay. See console for errors.');
    }
  };

  return (
    <div>
      <h1>WNBA Player Dashboard</h1>
      <p>Displaying player stats and recommendations.</p>

      {/* Button to test parlay creation */} 
      <button onClick={handleCreateParlay} style={{ marginBottom: '20px', padding: '10px' }}>
        Create Sample Parlay (Test)
      </button>

      <div style={{ marginBottom: '20px' }}>
        {loading && <p>Loading predictions...</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {!loading && !error && (
          <DataGrid predictions={predictions} /> // Pass predictions to DataGrid
        )}
      </div>
      
      <div>
        <h2>Performance Charts</h2>
        {/* TODO: Add controls to select player/stat for the chart */}
        {/* TODO: Potentially feed data from predictions or other sources to the chart */}
        <PlayerPerformanceChart playerName="Sample Player" />
      </div>
    </div>
  );
};

export default Dashboard; 