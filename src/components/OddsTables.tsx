import React, { useEffect, useState, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef } from 'ag-grid-community';
import { getGameOdds, getPlayerPropOdds, GameOdd, PlayerPropOdd } from '../services/api';

import 'ag-grid-community/styles/ag-theme-quartz.css';

const OddsTables: React.FC = () => {
  const [gameOdds, setGameOdds] = useState<GameOdd[]>([]);
  const [playerPropOdds, setPlayerPropOdds] = useState<PlayerPropOdd[]>([]);
  const [loadingGameOdds, setLoadingGameOdds] = useState<boolean>(true);
  const [loadingPlayerProps, setLoadingPlayerProps] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number>(1); // Default to player_id 1 for props

  useEffect(() => {
    const fetchGameOddsData = async () => {
      try {
        setLoadingGameOdds(true);
        const data = await getGameOdds();
        setGameOdds(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch game odds:", err);
        setError('Failed to load game odds.');
        setGameOdds([]);
      } finally {
        setLoadingGameOdds(false);
      }
    };
    fetchGameOddsData();
  }, []);

  useEffect(() => {
    if (!selectedPlayerId) return;
    const fetchPlayerPropOddsData = async () => {
      try {
        setLoadingPlayerProps(true);
        const data = await getPlayerPropOdds(selectedPlayerId);
        setPlayerPropOdds(data);
        setError(null);
      } catch (err) {
        console.error(`Failed to fetch player prop odds for player ${selectedPlayerId}:`, err);
        setError('Failed to load player prop odds.');
        setPlayerPropOdds([]);
      } finally {
        setLoadingPlayerProps(false);
      }
    };
    fetchPlayerPropOddsData();
  }, [selectedPlayerId]);

  const gameOddsColDefs = useMemo<ColDef<GameOdd>[]>(() => [
    { field: "game_id", headerName: "Game ID", filter: true, sortable: true, minWidth: 150 },
    { field: "home_team", headerName: "Home Team", filter: true, sortable: true, floatingFilter: true },
    { field: "away_team", headerName: "Away Team", filter: true, sortable: true, floatingFilter: true },
    { field: "home_team_odds", headerName: "Home Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "away_team_odds", headerName: "Away Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "spread", headerName: "Spread", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "over_under", headerName: "Over/Under", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "source", headerName: "Source", filter: true, sortable: true },
    { field: "last_updated", headerName: "Last Updated", sortable: true, filter: 'agDateColumnFilter' },
  ], []);

  const playerPropOddsColDefs = useMemo<ColDef<PlayerPropOdd>[]>(() => [
    { field: "prop_id", headerName: "Prop ID", filter: true, sortable: true, minWidth: 100 },
    { field: "player_name", headerName: "Player", filter: true, sortable: true, floatingFilter: true },
    // { field: "player_id", headerName: "Player ID", sortable: true, filter: 'agNumberColumnFilter' }, // Can be hidden if player_name is sufficient
    { field: "stat_type", headerName: "Market", filter: true, sortable: true },
    { field: "line", headerName: "Line", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "over_odds", headerName: "Over Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "under_odds", headerName: "Under Odds", sortable: true, filter: 'agNumberColumnFilter' },
    { field: "source", headerName: "Source", filter: true, sortable: true },
    { field: "last_updated", headerName: "Last Updated", sortable: true, filter: 'agDateColumnFilter' },
  ], []);

  if (error) {
    return <p style={{ color: 'red' }}>{error}</p>;
  }

  return (
    <div>
      <div style={{ marginBottom: '30px' }}>
        <h3>Game Odds</h3>
        {loadingGameOdds ? <p>Loading game odds...</p> :
          gameOdds.length === 0 ? <p>No game odds available.</p> :
          <div className="ag-theme-quartz" style={{ height: 400, width: '100%' }}>
            <AgGridReact<GameOdd>
              rowData={gameOdds}
              columnDefs={gameOddsColDefs}
              defaultColDef={{ flex: 1, minWidth: 120, resizable: true }}
              pagination={true} paginationPageSize={10} paginationPageSizeSelector={[10, 20, 50]}
            />
          </div>
        }
      </div>

      <div>
        <h3>Player Prop Odds</h3>
        <div style={{ marginBottom: '10px' }}>
          <label htmlFor="playerIdInput">Enter Player ID for Prop Odds: </label>
          <input 
            type="number" 
            id="playerIdInput" 
            value={selectedPlayerId}
            onChange={(e) => setSelectedPlayerId(parseInt(e.target.value, 10) || 0)} // Handle potential NaN
            style={{ padding: '5px', marginRight: '10px' }}
          />
          {/* Small note: Player ID 1 is used as default from mock data. Update as needed. */}
        </div>
        {loadingPlayerProps ? <p>Loading player prop odds for Player ID: {selectedPlayerId}...</p> :
          playerPropOdds.length === 0 ? <p>No player prop odds available for Player ID: {selectedPlayerId}.</p> :
          <div className="ag-theme-quartz" style={{ height: 400, width: '100%' }}>
            <AgGridReact<PlayerPropOdd>
              rowData={playerPropOdds}
              columnDefs={playerPropOddsColDefs}
              defaultColDef={{ flex: 1, minWidth: 120, resizable: true }}
              pagination={true} paginationPageSize={10} paginationPageSizeSelector={[10, 20, 50]}
            />
          </div>
        }
      </div>
    </div>
  );
};

export default OddsTables; 