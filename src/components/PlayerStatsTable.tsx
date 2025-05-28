import React, { useEffect, useState, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, ValueFormatterParams } from 'ag-grid-community';
import { getPlayerStats, PlayerStatFull } from '../services/api';

import 'ag-grid-community/styles/ag-grid.css'; // Core CSS - Changed path again
import 'ag-grid-community/styles/ag-theme-quartz.css'; // Theme

const PlayerStatsTable: React.FC = () => {
  const [playerStats, setPlayerStats] = useState<PlayerStatFull[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPlayerStats = async () => {
      try {
        setLoading(true);
        const data = await getPlayerStats();
        setPlayerStats(data);
        setError(null);
      } catch (err) {
        console.error("Failed to fetch player stats:", err);
        setError('Failed to load player statistics.');
        setPlayerStats([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPlayerStats();
  }, []);

  const colDefs = useMemo<ColDef<PlayerStatFull>[]>(() => [
    { field: "player.player_name", headerName: "Player", filter: true, sortable: true, floatingFilter: true },
    { field: "player.team_name", headerName: "Team", filter: true, sortable: true, floatingFilter: true },
    {
      field: "game.game_datetime",
      headerName: "Game DateTime",
      sortable: true,
      filter: 'agDateColumnFilter',
      valueFormatter: (params: ValueFormatterParams<PlayerStatFull, string | null | undefined>) => 
        params.value ? new Date(params.value).toLocaleString() : 'N/A',
    },
    { field: "minutes_played", headerName: "MIN", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "points", headerName: "PTS", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "rebounds", headerName: "REB", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "assists", headerName: "AST", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "steals", headerName: "STL", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "blocks", headerName: "BLK", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "turnovers", headerName: "TOV", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "field_goals_made", headerName: "FGM", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "field_goals_attempted", headerName: "FGA", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "three_pointers_made", headerName: "3PM", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "three_pointers_attempted", headerName: "3PA", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "free_throws_made", headerName: "FTM", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "free_throws_attempted", headerName: "FTA", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    { field: "plus_minus", headerName: "+/-", sortable: true, filter: 'agNumberColumnFilter', aggFunc: 'avg' },
    {
      headerName: "Matchup",
      valueGetter: (params): string => params.data ? `${params.data.game.away_team} @ ${params.data.game.home_team}` : 'N/A',
      filter: true, sortable: true, floatingFilter: true
    },
  ], []);

  if (loading) {
    return <p>Loading player statistics...</p>;
  }

  if (error) {
    return <p style={{ color: 'red' }}>{error}</p>;
  }

  if (playerStats.length === 0 && !loading) {
    return <p>No player statistics available.</p>;
  }

  return (
    <div style={{ marginBottom: '20px' }}>
      <h3>Player Statistics</h3>
      <div className="ag-theme-quartz" style={{ height: 500, width: '100%' }}>
        <AgGridReact<PlayerStatFull>
          rowData={playerStats}
          columnDefs={colDefs}
          defaultColDef={{
            flex: 1,
            minWidth: 100,
            resizable: true,
          }}
          pagination={true}
          paginationPageSize={15}
          paginationPageSizeSelector={[10, 15, 25, 50, 100]}
        />
      </div>
    </div>
  );
};

export default PlayerStatsTable; 