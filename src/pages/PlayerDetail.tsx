import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerStats, getPlayerPropOdds, PlayerStat, PlayerPropOdd } from '../services/api';
import PlayerPerformanceChart from '../components/Chart';
import { AgGridReact } from 'ag-grid-react';
import { ColDef } from 'ag-grid-community';

import 'ag-grid-community/styles/ag-theme-quartz.css';

const PlayerDetailPage: React.FC = () => {
  const { playerId } = useParams<{ playerId: string }>();
  const numericPlayerId = playerId ? parseInt(playerId, 10) : undefined;

  const [playerStats, setPlayerStats] = useState<PlayerStat[]>([]);
  const [playerInfo, setPlayerInfo] = useState<PlayerStat | null>(null); // To store common info like name/team
  const [propOdds, setPropOdds] = useState<PlayerPropOdd[]>([]);
  const [loadingStats, setLoadingStats] = useState<boolean>(true);
  const [loadingProps, setLoadingProps] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      if (!numericPlayerId && playerId !== undefined) { // Check if playerId is defined but not numeric
        setError(`Invalid player ID: ${playerId}`);
        setLoadingStats(false);
        setPlayerStats([]);
        return;
      }
      if (numericPlayerId === undefined) { // PlayerId is not in URL
        setLoadingStats(false);
        return;
      }
      try {
        setLoadingStats(true);
        const allStats = await getPlayerStats(); // Fetch all stats
        const filteredStats = allStats.filter(stat => stat.player_id === numericPlayerId || String(stat.player_id) === playerId);
        
        if (filteredStats.length > 0) {
          setPlayerStats(filteredStats);
          setPlayerInfo(filteredStats[0]); // Set common player info from the first stat entry
        } else {
          setPlayerStats([]);
          setPlayerInfo(null);
          // setError(prev => prev ? `${prev}\nNo stats found for player ID: ${playerId}` : `No stats found for player ID: ${playerId}`);
        }
        // setError(null); // Clear previous general errors if this part succeeds
      } catch (err) {
        console.error(`Failed to fetch player stats for ID ${playerId}:`, err);
        setError(prev => prev ? `${prev}\nFailed to load stats for player ID: ${playerId}` : `Failed to load stats for player ID: ${playerId}`);
        setPlayerStats([]);
        setPlayerInfo(null);
      } finally {
        setLoadingStats(false);
      }
    };

    fetchStats();
  }, [playerId, numericPlayerId]);

  useEffect(() => {
    if (!numericPlayerId) {
        setLoadingProps(false);
        return;
    }
    const fetchProps = async () => {
      try {
        setLoadingProps(true);
        const propsData = await getPlayerPropOdds(numericPlayerId);
        setPropOdds(propsData);
        // setError(null); // Clear previous general errors if this part succeeds
      } catch (err) {
        console.error(`Failed to fetch prop odds for player ID ${numericPlayerId}:`, err);
        setError(prev => prev ? `${prev}\nFailed to load prop odds for player ID: ${numericPlayerId}`: `Failed to load prop odds for player ID: ${numericPlayerId}`);
        setPropOdds([]);
      } finally {
        setLoadingProps(false);
      }
    };

    fetchProps();
  }, [numericPlayerId]);

  const statsColDefs = useMemo<ColDef<PlayerStat>[]>(() => [
    { field: "game_date", headerName: "Date", sortable: true, filter: 'agDateColumnFilter', valueFormatter: p => p.value ? new Date(p.value).toLocaleDateString() : '' },
    { field: "points", headerName: "PTS", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "rebounds", headerName: "REB", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "assists", headerName: "AST", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "steals", headerName: "STL", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "blocks", headerName: "BLK", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "turnovers", headerName: "TOV", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "minutes_played", headerName: "MIN", sortable: true, filter: 'agNumberColumnFilter' },
  ], []);

 const propOddsColDefs = useMemo<ColDef<PlayerPropOdd>[]>(() => [
    { field: "stat_type", headerName: "Market", filter: true, sortable: true },
    { field: "line", headerName: "Line", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "over_odds", headerName: "Over Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "under_odds", headerName: "Under Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "source", headerName: "Source", filter: true, sortable: true },
    { field: "last_updated", headerName: "Last Updated", sortable: true, filter: 'agDateColumnFilter', valueFormatter: p => p.value ? new Date(p.value).toLocaleString() : '' },
  ], []);

  const isLoading = loadingStats || loadingProps;

  if (isLoading) {
    return <p>Loading player details for ID: {playerId}...</p>;
  }

  if (error) {
    return <p style={{ color: 'red' }}>{error.split('\n').map((line, idx) => <span key={idx}>{line}<br/></span>)}</p>;
  }

  if (!playerInfo && playerStats.length === 0 && !isLoading) {
    return <p>No player found with ID: {playerId}. Please check the ID and try again.</p>;
  }

  return (
    <div>
      {playerInfo && (
        <>
          <h1>{playerInfo.player_name}</h1>
          <p><strong>Team:</strong> {playerInfo.team_name}</p>
          <p><strong>Player ID:</strong> {numericPlayerId}</p>
        </>
      )}
      {!playerInfo && <h1>Player Detail Page</h1>} {/* Fallback title */}

      <div style={{ marginTop: '20px', marginBottom: '30px' }}>
        <h2>Performance Chart</h2>
        {playerStats.length > 0 && playerInfo ? (
          <PlayerPerformanceChart 
            playerName={playerInfo.player_name} 
            playerStats={playerStats} 
            statToDisplay="points" // Default to points, can be made dynamic later
          />
        ) : (
          <p>No game statistics available to display chart for {playerInfo?.player_name || `player ID ${playerId}`}.</p>
        )}
      </div>

      <div style={{ marginBottom: '30px' }}>
        <h2>Game Statistics</h2>
        {playerStats.length > 0 ? (
          <div className="ag-theme-quartz" style={{ height: 400, width: '100%' }}>
            <AgGridReact<PlayerStat>
              rowData={playerStats}
              columnDefs={statsColDefs}
              defaultColDef={{ flex: 1, minWidth: 100, resizable: true }}
              pagination={true} paginationPageSize={10} paginationPageSizeSelector={[5, 10, 20]}
            />
          </div>
        ) : (
          <p>No game statistics found for this player.</p>
        )}
      </div>

      <div>
        <h2>Player Prop Odds</h2>
        {propOdds.length > 0 ? (
          <div className="ag-theme-quartz" style={{ height: 300, width: '100%' }}>
            <AgGridReact<PlayerPropOdd>
              rowData={propOdds}
              columnDefs={propOddsColDefs}
              defaultColDef={{ flex: 1, minWidth: 120, resizable: true }}
              pagination={true} paginationPageSize={5} paginationPageSizeSelector={[5, 10]}
            />
          </div>
        ) : (
          <p>No current prop odds found for this player.</p>
        )}
      </div>
    </div>
  );
};

export default PlayerDetailPage; 