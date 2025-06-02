import React, { useEffect, useState, useMemo } from 'react';
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
  const [selectedPlayerIdForChart, setSelectedPlayerIdForChart] = useState<string | null>(null); // Added state for selected player ID
  const [selectedDateRangeFilter, setSelectedDateRangeFilter] = useState<string>('all'); // Added state for date range filter
  const [selectedLocationFilter, setSelectedLocationFilter] = useState<string>('all'); // Added state for home/away filter
  const [selectedOpponentTeamNameForH2H, setSelectedOpponentTeamNameForH2H] = useState<string>('all'); // Added state for H2H opponent
  const [playerSearchTerm, setPlayerSearchTerm] = useState<string>(''); // Added state for player search
  const [selectedSeasonForChart, setSelectedSeasonForChart] = useState<string>('all'); // 'all' or a specific season year as string

  // Calculate combined probability for the staged parlay
  const combinedParlayProbability = useMemo(() => {
    if (stagedParlayLegs.length === 0) {
      return null;
    }
    let probability = 1;
    for (const leg of stagedParlayLegs) {
      if (leg.type === 'over') {
        probability *= (leg.prediction.predicted_over_probability ?? 0);
      } else {
        probability *= (leg.prediction.predicted_under_probability ?? 0);
      }
    }
    return probability;
  }, [stagedParlayLegs]);

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
        const statsData = await getPlayerStats();
        setAllPlayerStats(statsData);
        // console.log('All Player Stats Fetched:', statsData); // Log can be removed or kept for debugging
        if (statsData.length > 0 && !selectedPlayerIdForChart) {
          setSelectedPlayerIdForChart(statsData[0].player_id);
        }
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
  const uniquePlayersForChart = useMemo(() => {
    const playerMap = new Map<string, { id: string; name: string }>();
    allPlayerStats.forEach(stat => {
      if (stat.player && !playerMap.has(stat.player_id)) {
        playerMap.set(stat.player_id, { id: stat.player_id, name: stat.player.player_name });
      }
    });
    return Array.from(playerMap.values());
  }, [allPlayerStats]);

  const uniqueSeasonsForChart = useMemo(() => {
    const seasons = new Set<string>();
    allPlayerStats.forEach(stat => {
      if (stat.game_date) {
        seasons.add(new Date(stat.game_date).getFullYear().toString());
      }
    });
    return ['all', ...Array.from(seasons).sort((a, b) => parseInt(b) - parseInt(a))]; // 'all' plus sorted years
  }, [allPlayerStats]);

  const filteredUniquePlayersForChart = useMemo(() => {
    if (!playerSearchTerm) {
      return uniquePlayersForChart;
    }
    return uniquePlayersForChart.filter(player =>
      player.name.toLowerCase().includes(playerSearchTerm.toLowerCase())
    );
  }, [uniquePlayersForChart, playerSearchTerm]);

  const selectedPlayerForChart = useMemo(() => {
    if (!selectedPlayerIdForChart) return null;
    return filteredUniquePlayersForChart.find(p => p.id === selectedPlayerIdForChart) || null;
  }, [selectedPlayerIdForChart, filteredUniquePlayersForChart]);
  
  const chartPlayerName = selectedPlayerForChart ? selectedPlayerForChart.name : "Select a Player";
  const chartPlayerStats = useMemo(() => {
    let stats = allPlayerStats.filter(stat => stat.player_id === selectedPlayerIdForChart);
    
    // Apply Season Filter first
    if (selectedSeasonForChart !== 'all') {
      stats = stats.filter(stat => stat.game_date && new Date(stat.game_date).getFullYear().toString() === selectedSeasonForChart);
    }

    // Apply H2H filter if an opponent is selected
    if (selectedPlayerForChart && selectedOpponentTeamNameForH2H !== 'all') {
        stats = stats.filter(stat => {
            if (!stat.game || !stat.player.team_name) return false;
            const playerTeam = stat.player.team_name;
            const homeTeam = stat.game.home_team;
            const awayTeam = stat.game.away_team;
            // Game is H2H if player's team is home and opponent is away, OR player's team is away and opponent is home
            return (homeTeam === playerTeam && awayTeam === selectedOpponentTeamNameForH2H) || 
                   (awayTeam === playerTeam && homeTeam === selectedOpponentTeamNameForH2H);
        });
    }
    
    // Apply Home/Away filter first (as it's independent of date sorting for slicing)
    if (selectedPlayerForChart && selectedLocationFilter !== 'all' && selectedOpponentTeamNameForH2H === 'all') {
      stats = stats.filter(stat => {
        if (!stat.game || !stat.player.team_name) return false;
        const playerTeamName = stat.player.team_name;
        if (selectedLocationFilter === 'home') {
          return stat.game.home_team === playerTeamName;
        }
        if (selectedLocationFilter === 'away') {
          return stat.game.away_team === playerTeamName;
        }
        return true; // Should not happen if selectedLocationFilter is 'home' or 'away'
      });
    }
    
    // Sort by game_date descending to easily get last N games
    stats.sort((a, b) => new Date(b.game_date as string).getTime() - new Date(a.game_date as string).getTime());

    switch (selectedDateRangeFilter) {
      case 'last5':
        stats = stats.slice(0, 5);
        break;
      case 'last10':
        stats = stats.slice(0, 10);
        break;
      case 'last20':
        stats = stats.slice(0, 20);
        break;
      case 'entireSeason': // Renamed from 'all' for clarity when season is selected
        // If a specific season is selected, 'entireSeason' means all games of that season.
        // If 'all' seasons is selected, this effectively means 'all time' if not further sliced by 'allTime' option below.
        // No slicing here, the season filter above handles it.
        break;
      case 'allTime':
        // No slicing, shows all games (respecting other filters like player, location, H2H, and selectedSeason if not 'all')
        break;
      default:
        // Default to 'allTime' or 'entireSeason' if a season is chosen.
        // This case should ideally not be hit if selectedDateRangeFilter is managed.
        break;
    }
    // The chart component expects data sorted ascendingly for the time axis
    return stats.sort((a, b) => new Date(a.game_date as string).getTime() - new Date(b.game_date as string).getTime());
  }, [allPlayerStats, selectedPlayerIdForChart, selectedDateRangeFilter, selectedLocationFilter, selectedOpponentTeamNameForH2H, selectedSeasonForChart]);

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

    const parlaySelectionsData = stagedParlayLegs.map(leg => {
      const pred = leg.prediction;
      const prop = pred.player_prop;
      let chosenProb = 0;
      if (leg.type === 'over' && pred.predicted_over_probability !== null) {
        chosenProb = pred.predicted_over_probability;
      } else if (leg.type === 'under' && pred.predicted_under_probability !== null) {
        chosenProb = pred.predicted_under_probability;
      }

      return {
        prediction_id: pred.id,
        player_prop_id: prop?.id || 'N/A', // Provide a fallback or ensure this is always available
        player_name: prop?.player?.player_name || 'N/A',
        market_key: prop?.market?.key || 'N/A',
        game_id: prop?.game_id || 'N/A', 
        line_point: prop?.outcomes?.find(o => o.point !== undefined)?.point ?? null, // Ensure this matches backend Optional[float]
        chosen_outcome: leg.type, // 'over' or 'under' directly
        chosen_probability: chosenProb,
      };
    });

    // Calculate combined probability for the backend payload if needed, or let backend calculate it
    // For now, sending what the frontend calculated. Backend might recalculate/override.
    const currentCombinedProbability = combinedParlayProbability;

    const newParlayPayload = {
      selections: parlaySelectionsData,
      combined_probability: currentCombinedProbability, // Sending the frontend calculated one
      // total_odds: undefined, // Let backend calculate if not available or relevant from frontend
    };

    try {
      console.log("Attempting to post parlay with payload:", newParlayPayload);
      // Ensure postParlay is called with this new structure, 
      // and ParlayData in api.ts matches this structure if we want strict typing there too.
      const result = await postParlay(newParlayPayload);
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

  const dateRangeFilterOptions = [
    { value: 'allTime', label: 'All Time' },
    { value: 'entireSeason', label: 'Selected Season' }, // Label changed
    { value: 'last5', label: 'Last 5 Games (in Season)' },
    { value: 'last10', label: 'Last 10 Games (in Season)' },
    { value: 'last20', label: 'Last 20 Games (in Season)' },
  ];

  const locationFilterOptions = [
    { value: 'all', label: 'All Games' },
    { value: 'home', label: 'Home Only' }, // Simplified label
    { value: 'away', label: 'Away Only' }, // Simplified label
  ];

  const uniqueOpponentTeamNames = useMemo(() => {
    const teamNames = new Set<string>();
    allPlayerStats.forEach(stat => {
      if (stat.game) {
        if (stat.game.home_team && stat.game.home_team !== stat.player.team_name) {
          teamNames.add(stat.game.home_team);
        }
        if (stat.game.away_team && stat.game.away_team !== stat.player.team_name) {
          teamNames.add(stat.game.away_team);
        }
      }
    });
    return Array.from(teamNames).sort();
  }, [allPlayerStats]);

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
          <div style={{ width: '100%', marginBottom: '20px' }}>
            <h2>Predictions</h2>
            <p style={{ fontSize: '0.9em', color: '#DDD', marginBottom: '10px' }}>
              Select predictions from the table below to add them to the Parlay Builder. 
              You can choose 'Over' or 'Under' for each leg in the builder section that appears.
            </p>
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
                  {stagedParlayLegs.map(leg => {
                    const prediction = leg.prediction;
                    const playerProp = prediction.player_prop;
                    const playerName = playerProp?.player?.player_name || 'N/A';
                    const market = playerProp?.market?.key || 'N/A';
                    // Attempt to find the line from outcomes, assuming it might be there if not directly on player_prop
                    // This depends on your data structure for player_prop.outcomes
                    const line = playerProp?.outcomes?.find(o => o.point !== undefined)?.point ?? playerProp?.outcomes?.[0]?.point ?? 'N/A';
                    const displayMarket = playerProp?.market?.description || market;

                    return (
                      <li key={prediction.id} style={{ marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px solid #eee'}}>
                        <strong>{playerName}</strong> - {displayMarket}: {line}
                        <br />
                        Selected: {leg.type === 'over' ? 'Over' : 'Under'}
                        {/* Display individual leg probability */}
                        (Prob: {leg.type === 'over' ? 
                          (prediction.predicted_over_probability !== null ? (prediction.predicted_over_probability * 100).toFixed(1) + '%' : 'N/A') : 
                          (prediction.predicted_under_probability !== null ? (prediction.predicted_under_probability * 100).toFixed(1) + '%' : 'N/A')
                        })
                        <div style={{ marginTop: '5px' }}>
                          <label style={{ marginRight: '10px'}}>
                            <input 
                              type="radio" 
                              name={`parlayLegType-${prediction.id}`} 
                              value="over" 
                              checked={leg.type === 'over'} 
                              onChange={() => handleParlayLegTypeChange(prediction.id, 'over')} 
                            /> Over
                          </label>
                          <label>
                            <input 
                              type="radio" 
                              name={`parlayLegType-${prediction.id}`} 
                              value="under" 
                              checked={leg.type === 'under'} 
                              onChange={() => handleParlayLegTypeChange(prediction.id, 'under')} 
                            /> Under
                          </label>
                        </div>
                      </li>
                    );
                  })}
                </ul>
                {combinedParlayProbability !== null && (
                  <p style={{ marginTop: '10px', fontWeight: 'bold' }}>
                    Combined Parlay Probability: {(combinedParlayProbability * 100).toFixed(2)}%
                  </p>
                )}
                <button onClick={handleCreateParlayFromStaged} style={{ marginTop: '10px' }}>
                  Create Parlay with {stagedParlayLegs.length} Leg(s)
                </button>
                {/* TODO: Add UI here to select 'over'/'under' for each staged leg */}
              </div>
            )}
          </div>
          
          <div style={{ width: '100%' }}>
            <h2>Performance Charts</h2>
            {chartPlayerStats.length > 0 || selectedPlayerIdForChart ? (
              <>
                <div style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
                  <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                    <label htmlFor="player-select-chart" style={{ marginRight: '10px' }}>Player:</label>
                    <input
                      type="text"
                      placeholder="Search player..."
                      value={playerSearchTerm}
                      onChange={(e) => setPlayerSearchTerm(e.target.value)}
                      style={{ marginRight: '10px', padding: '5px' }}
                    />
                    <select
                      id="player-select-chart"
                      value={selectedPlayerIdForChart || ''}
                      onChange={(e) => setSelectedPlayerIdForChart(e.target.value || null)}
                    >
                      <option value="" disabled>{playerSearchTerm ? 'No matches' : 'Select a Player'}</option>
                      {filteredUniquePlayersForChart.map(player => (
                        <option key={player.id} value={player.id}>
                          {player.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ marginRight: '20px', marginBottom: '10px' }}>
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
                  <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                    <label htmlFor="season-select-chart" style={{ marginRight: '10px' }}>Season:</label>
                    <select
                      id="season-select-chart"
                      value={selectedSeasonForChart}
                      onChange={(e) => {
                        setSelectedSeasonForChart(e.target.value);
                        // If a specific season is chosen, 'All Time' might be confusing for date range,
                        // so default to 'Selected Season' (entire season)
                        if (e.target.value !== 'all') {
                          setSelectedDateRangeFilter('entireSeason');
                        } else {
                          setSelectedDateRangeFilter('allTime'); // Default to allTime if "All Seasons" is picked
                        }
                      }}
                    >
                      {uniqueSeasonsForChart.map(season => (
                        <option key={season} value={season}>
                          {season === 'all' ? 'All Seasons' : season}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ marginBottom: '10px' }}>
                    <label htmlFor="date-range-select" style={{ marginRight: '10px' }}>Game Range:</label>
                    <select
                      id="date-range-select"
                      value={selectedDateRangeFilter}
                      onChange={(e) => setSelectedDateRangeFilter(e.target.value)}
                    >
                      {dateRangeFilterOptions.map(option => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ marginBottom: '10px' }}>
                    <label htmlFor="location-filter-select" style={{ marginRight: '10px' }}>Location:</label>
                    <select
                      id="location-filter-select"
                      value={selectedLocationFilter}
                      onChange={(e) => setSelectedLocationFilter(e.target.value)}
                    >
                      {locationFilterOptions.map(option => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div style={{ marginBottom: '10px' }}>
                    <label htmlFor="h2h-opponent-select" style={{ marginRight: '10px' }}>Vs Opponent:</label>
                    <select
                      id="h2h-opponent-select"
                      value={selectedOpponentTeamNameForH2H}
                      onChange={(e) => {
                        setSelectedOpponentTeamNameForH2H(e.target.value);
                        // Optionally reset location filter if H2H is selected, or manage interaction
                        if (e.target.value !== 'all') {
                          setSelectedLocationFilter('all'); // Reset location if H2H is chosen
                        }
                      }}
                    >
                      <option value="all">All Opponents</option>
                      {uniqueOpponentTeamNames.map(teamName => (
                        <option key={teamName} value={teamName}>
                          {teamName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <PlayerPerformanceChart 
                  playerName={chartPlayerName} 
                  playerStats={chartPlayerStats} 
                  statToDisplay={selectedChartStat} // Use state here
                />
              </>
            ) : (
              <p>No data available for chart {selectedPlayerForChart ? `for ${selectedPlayerForChart.name}` : ''}. Select a player to view their chart.</p>
            )}
          </div>

          {/* Display Player Statistics Table */}
          {/* PlayerStatsTable fetches its own data, so no need to pass allPlayerStats if it handles its own state */}
          <div style={{ width: '100%', marginBottom: '20px' }}>
            <PlayerStatsTable />
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard; 