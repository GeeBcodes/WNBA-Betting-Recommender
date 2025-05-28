import React, { useEffect, useState } from 'react';
import { getParlays, postParlay, Parlay, ParlayData, getPredictions, Prediction } from '../services/api';

const ParlayGeneratorPage: React.FC = () => {
  const [parlays, setParlays] = useState<Parlay[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]); // To select legs for a new parlay
  const [loadingParlays, setLoadingParlays] = useState<boolean>(true);
  const [loadingPredictions, setLoadingPredictions] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoadingParlays(true);
        setLoadingPredictions(true);
        const [parlaysData, predictionsData] = await Promise.all([
          getParlays(),
          getPredictions() // Fetch predictions to use for creating new parlays
        ]);
        setParlays(parlaysData);
        setPredictions(predictionsData);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch parlay data:", err);
        setError('Failed to load parlay information. Please try again later.');
        setParlays([]);
        setPredictions([]);
      } finally {
        setLoadingParlays(false);
        setLoadingPredictions(false);
      }
    };

    fetchData();
  }, []);

  const handleCreateSampleParlay = async () => {
    if (predictions.length < 2) {
      alert("Not enough predictions available to create a sample parlay (need at least 2).");
      return;
    }

    const sampleParlayData: ParlayData = {
      name: `Sample Parlay ${new Date().toLocaleTimeString()}`,
      legs: [
        { prediction_id: predictions[0].id, type: 'over' as const },
        { prediction_id: predictions[1].id, type: 'under' as const },
      ],
    };

    try {
      console.log("Attempting to post parlay:", sampleParlayData);
      const newParlay = await postParlay(sampleParlayData);
      console.log('Parlay created successfully:', newParlay);
      alert('Sample Parlay posted! Check the console for details. Refreshing parlay list...');
      // Refresh parlays list
      setLoadingParlays(true);
      const updatedParlays = await getParlays();
      setParlays(updatedParlays);
      setLoadingParlays(false);
    } catch (err) {
      console.error('Failed to post parlay:', err);
      alert('Failed to post sample parlay. See console for errors.');
    }
  };

  if (loadingParlays || loadingPredictions) {
    return <p>Loading parlay generator...</p>;
  }

  if (error) {
    return <p style={{ color: 'red' }}>{error}</p>;
  }

  return (
    <div>
      <h1>Parlay Generator</h1>
      
      <button onClick={handleCreateSampleParlay} style={{ marginBottom: '20px', padding: '10px' }} disabled={predictions.length < 2}>
        Create Sample Parlay
      </button>

      <h2>Existing Parlays</h2>
      {parlays.length === 0 ? (
        <p>No parlays found.</p>
      ) : (
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {parlays.map((parlay) => (
            <li key={parlay.id} style={{ border: '1px solid #eee', padding: '10px', marginBottom: '10px', borderRadius: '4px' }}>
              <p><strong>ID:</strong> {parlay.id}</p>
              <p><strong>Created:</strong> {new Date(parlay.created_at).toLocaleString()}</p>
              <p><strong>Selections:</strong></p>
              <pre style={{ backgroundColor: '#f0f0f0', padding: '5px', borderRadius: '3px' }}>
                {JSON.stringify(parlay.selections, null, 2)}
              </pre>
              {parlay.combined_probability && <p><strong>Combined Probability:</strong> {(parlay.combined_probability * 100).toFixed(2)}%</p>}
              {parlay.total_odds && <p><strong>Total Odds:</strong> {parlay.total_odds}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ParlayGeneratorPage; 