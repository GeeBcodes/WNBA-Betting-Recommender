import React, { useEffect, useState } from 'react';
import DataGrid from '../components/DataGrid';
import PlayerPerformanceChart from '../components/Chart';
import ModelVersionInfo from '../components/ModelVersionInfo';
import PlayerStatsTable from '../components/PlayerStatsTable';
import { getPredictions, /* postParlay, */ Prediction, /* ParlayData, */ PlayerStatFull, getPlayerStats, postParlay, ParlayData } from '../services/api'; // Import PlayerStatFull and getPlayerStats

// Interface for a staged parlay leg
export interface StagedParlayLeg {
  prediction: Prediction;
  type: 'over' | 'under';
}

const Dashboard = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [allPlayerStats, setAllPlayerStats] = useState<PlayerStatFull[]>([]); // State for all player stats, updated type
  const [stagedParlayLegs, setStagedParlayLegs] = useState<StagedParlayLeg[]>([]);
  const [loadingPredictions, setLoadingPredictions] = useState<boolean>(true);
  const [loadingPlayerStats, setLoadingPlayerStats] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showModelInfoPopover, setShowModelInfoPopover] = useState<boolean>(false);
  const infoIconRef = React.useRef<HTMLSpanElement>(null);
  const [selectedChartStat, setSelectedChartStat] = useState<keyof PlayerStatFull>("points"); // Default to 'points', updated type

  // Fetch predictions
  useEffect(() => {
    const fetchPredictionsData = async () => {
      try {
        setLoadingPredictions(true);
        const data = await getPredictions();
        setPredictions(data);
        // setError(null); // Consolidate error handling
      } catch (err) {
        console.error("Failed to fetch predictions:", err);
        setError(prevError => prevError ? `${prevError}\nFailed to load predictions.` : 'Failed to load predictions.');
        setPredictions([]);
      } finally {
        setLoadingPredictions(false);
      }
    };
    fetchPredictionsData();
  }, []);

  // Fetch all player stats
  useEffect(() => {
    const fetchAllPlayerStats = async () => {
      try {
        setLoadingPlayerStats(true);
        const statsData = await getPlayerStats(); // Use the new getPlayerStats from api.ts
        setAllPlayerStats(statsData); // Type matches now
        // setError(null); // Consolidate error handling
      } catch (err) {
        console.error("Failed to fetch player stats:", err);
        setError(prevError => prevError ? `${prevError}\nFailed to load player stats.` : 'Failed to load player stats.');
        setAllPlayerStats([]);
      } finally {
        setLoadingPlayerStats(false);
      }
    };
    fetchAllPlayerStats();
  }, []);

  // Derived state for the chart
  const chartPlayerName = allPlayerStats.length > 0 && allPlayerStats[0].player ? allPlayerStats[0].player.player_name : "Sample Player"; // Updated access
  const chartPlayerStats = allPlayerStats.filter(stat => stat.player && stat.player.player_name === chartPlayerName); // Updated access

  const handlePredictionSelectionChange = (selectedRows: Prediction[]) => {
    // When selection changes, map selected predictions to StagedParlayLeg objects
    // Default to 'over', or try to preserve existing type if the prediction was already staged
    const newStagedLegs = selectedRows.map(selectedPred => {
      const existingStagedLeg = stagedParlayLegs.find(leg => leg.prediction.id === selectedPred.id);
      return {
        prediction: selectedPred,
        type: existingStagedLeg ? existingStagedLeg.type : 'over', // Default to 'over' for new selections
      };
    });
    setStagedParlayLegs(newStagedLegs);
  };

  const handleParlayLegTypeChange = (predictionId: string, newType: 'over' | 'under') => {
    setStagedParlayLegs(prevLegs => 
      prevLegs.map(leg => 
        leg.prediction.id === predictionId ? { ...leg, type: newType } : leg
      )
    );
  };

  const handleCreateParlayFromStaged = async () => {
    if (stagedParlayLegs.length === 0) {
      alert("Please select at least one prediction from the table to create a parlay.");
      return;
    }
    // For now, we'll assume all selected legs are 'over' by default.
    // This should be enhanced to allow users to specify 'over' or 'under' for each leg.
    const parlayLegsData = stagedParlayLegs.map(leg => ({ 
      prediction_id: leg.prediction.id, 
      type: leg.type // Use the user-selected type
    }));

    const newParlayData: ParlayData = {
      name: `Dashboard Parlay ${new Date().toLocaleTimeString()}`,
      legs: parlayLegsData,
    };

    try {
      console.log("Attempting to post parlay from staged legs:", newParlayData);
      const result = await postParlay(newParlayData);
      console.log('Parlay created successfully:', result);
      alert('Parlay created from selected predictions! Check console.');
      setStagedParlayLegs([]); // Clear staged legs after successful parlay creation
      // Optionally, refresh parlay list on ParlayGeneratorPage or here if displayed
    } catch (err) {
      console.error('Failed to post parlay from staged legs:', err);
      alert('Failed to create parlay. See console for errors.');
    }
  };

  // Overall loading and error display
  const isLoading = loadingPredictions || loadingPlayerStats;

  const availableStats: Array<{ value: keyof PlayerStatFull; label: string }> = [ // Updated type
    { value: "points", label: "Points" },
    { value: "rebounds", label: "Rebounds" },
    { value: "assists", label: "Assists" },
    { value: "steals", label: "Steals" },
    { value: "blocks", label: "Blocks" },
    { value: "turnovers", label: "Turnovers" },
    { value: "minutes_played", label: "Minutes Played" },
    // Add other relevant stats from PlayerStatFull if needed for the chart dropdown
    { value: "field_goals_made", label: "FG Made" },
    { value: "field_goals_attempted", label: "FG Attempted" },
    { value: "three_pointers_made", label: "3P Made" },
    { value: "three_pointers_attempted", label: "3P Attempted" },
    { value: "free_throws_made", label: "FT Made" },
    { value: "free_throws_attempted", label: "FT Attempted" },
    { value: "plus_minus", label: "Plus/Minus" },
  ];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
        <h1>WNBA Player Dashboard</h1>
        <span 
          ref={infoIconRef} // Add ref to the icon
          onMouseEnter={() => setShowModelInfoPopover(true)}
          // onMouseLeave={() => setShowModelInfoPopover(false)}
          style={{
            marginLeft: '15px', 
            cursor: 'pointer', 
            fontSize: '1.5em', 
            position: 'relative' // Keep relative for potential child absolute positioning
          }}
          title="View Model Version Info"
        >
          ⓘ
        </span>
        {showModelInfoPopover && infoIconRef.current && (
          <div 
            onMouseLeave={() => setShowModelInfoPopover(false)} // Hide when mouse leaves popover
            style={{
              position: 'absolute',
              // Position relative to the icon
              top: `${infoIconRef.current.offsetTop + infoIconRef.current.offsetHeight + 5}px`, // Below the icon + 5px margin
              left: `${infoIconRef.current.offsetLeft}px`, // Aligned with the icon's left
              backgroundColor: 'white',
              border: '1px solid #ccc',
              borderRadius: '5px',
              padding: '0px', // ModelVersionInfo has its own padding
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
              zIndex: 1000, 
              minWidth: '300px' 
            }}
          >
            <ModelVersionInfo />
          </div>
        )}
      </div>
      <p>Displaying player stats and recommendations.</p>

      {isLoading && <p>Loading dashboard data...</p>}
      {error && <p style={{ color: 'red' }}>{error.split('\n').map((line, idx) => <span key={idx}>{line}<br/></span>)}</p>}

      {!isLoading && !error && (
        <>
          {/* Display Player Statistics Table */}
          {/* PlayerStatsTable fetches its own data, so no need to pass allPlayerStats if it handles its own state */}
          <div style={{ width: '100%', marginBottom: '20px' }}>
            <PlayerStatsTable />
          </div>

          <div style={{ width: '100%', marginBottom: '20px' }}>
            <h2>Predictions</h2>
            {predictions.length === 0 ? (
              <p>No predictions available.</p>
            ) : (
              <DataGrid predictions={predictions} onSelectionChanged={handlePredictionSelectionChange} />
            )}
            {/* Display selected predictions for parlay building */} 
            {stagedParlayLegs.length > 0 && (
              <div style={{ marginTop: '15px', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
                <h4>Parlay Builder - Selected Predictions:</h4>
                <ul>
                  {stagedParlayLegs.map(leg => (
                    <li key={leg.prediction.id} style={{ marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px solid #eee'}}>
                      Prediction ID: {leg.prediction.id} (Prop: {leg.prediction.player_prop_odd_id})
                      <div style={{ marginTop: '5px' }}>
                        <label style={{ marginRight: '10px'}}>
                          <input 
                            type="radio" 
                            name={`parlayLegType-${leg.prediction.id}`} 
                            value="over" 
                            checked={leg.type === 'over'} 
                            onChange={() => handleParlayLegTypeChange(leg.prediction.id, 'over')} 
                          /> Over
                        </label>
                        <label>
                          <input 
                            type="radio" 
                            name={`parlayLegType-${leg.prediction.id}`} 
                            value="under" 
                            checked={leg.type === 'under'} 
                            onChange={() => handleParlayLegTypeChange(leg.prediction.id, 'under')} 
                          /> Under
                        </label>
                      </div>
                    </li>
                  ))}
                </ul>
                <button onClick={handleCreateParlayFromStaged} style={{ marginTop: '10px' }}>
                  Create Parlay with {stagedParlayLegs.length} Leg(s)
                </button>
                {/* TODO: Add UI here to select 'over'/'under' for each staged leg */}
              </div>
            )}
          </div>
          
          <div style={{ width: '100%' }}>
            <h2>Performance Charts</h2>
            {chartPlayerStats.length > 0 ? (
              <>
                <div style={{ marginBottom: '10px' }}>
                  <label htmlFor="stat-select" style={{ marginRight: '10px' }}>Displaying Stat:</label>
                  <select 
                    id="stat-select" 
                    value={selectedChartStat} 
                    onChange={(e) => setSelectedChartStat(e.target.value as keyof PlayerStatFull)} // Updated type cast
                  >
                    {availableStats.map(statOption => (
                      <option key={statOption.value} value={statOption.value}>
                        {statOption.label}
                      </option>
                    ))}
                  </select>
                </div>
                <PlayerPerformanceChart 
                  playerName={chartPlayerName} 
                  playerStats={chartPlayerStats} 
                  statToDisplay={selectedChartStat} // Use state here
                />
              </>
            ) : (
              <p>No data available for chart for {chartPlayerName}.</p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard; 