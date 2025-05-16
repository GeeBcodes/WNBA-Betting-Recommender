import React, { useState, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef } from 'ag-grid-community';

// Placeholder for WNBA Player Stats Row Data
interface PlayerStatsRow {
  playerName: string;
  team: string;
  opponent: string;
  market: string; // e.g., Points, Rebounds, Assists
  line: number;
  overProbability?: number;
  underProbability?: number;
  pastPerformance?: string; // Could be a string like "W-L-W-W-L" or avg stats
  altLines?: string; // e.g., "O 10.5 (-110), U 10.5 (-110)"
}

const DataGrid: React.FC = () => {
  // Row Data: The data to be displayed.
  // TODO: This will eventually come from props or a data fetching hook
  const [rowData, setRowData] = useState<PlayerStatsRow[]>([
    { playerName: "Arike Ogunbowale", team: "DAL", opponent: "LVA", market: "Points", line: 22.5, overProbability: 0.55, pastPerformance: "25, 20, 30", altLines: "O 20.5, U 24.5" },
    { playerName: "Breanna Stewart", team: "NYL", opponent: "CON", market: "Rebounds", line: 9.5, overProbability: 0.60, pastPerformance: "10, 8, 12", altLines: "O 8.5, U 10.5" },
    { playerName: "Alyssa Thomas", team: "CON", opponent: "NYL", market: "Assists", line: 7.5, overProbability: 0.50, pastPerformance: "8, 7, 9", altLines: "O 6.5, U 8.5" },
  ]);

  // Column Definitions: Defines & controls grid columns.
  // TODO: This might also come from props or be more configurable
  const colDefs = useMemo<ColDef<PlayerStatsRow>[]>(() => [
    { field: "playerName", headerName: "Player", filter: true, sortable: true },
    { field: "team", headerName: "Team", filter: true, sortable: true },
    { field: "opponent", headerName: "Opponent", filter: true, sortable: true },
    { field: "market", headerName: "Market", filter: true, sortable: true },
    { field: "line", headerName: "Line", sortable: true, cellDataType: 'number' },
    { field: "overProbability", headerName: "Over Prob.", sortable: true, valueFormatter: p => p.value ? (p.value * 100).toFixed(1) + '%' : '', cellDataType: 'number' },
    { field: "underProbability", headerName: "Under Prob.", sortable: true, valueFormatter: p => p.value ? (p.value * 100).toFixed(1) + '%' : '', cellDataType: 'number' },
    { field: "pastPerformance", headerName: "Past Performance", minWidth: 150 },
    { field: "altLines", headerName: "Alt Lines", minWidth: 150 },
  ], []);

  return (
    <div className="ag-theme-quartz" style={{ height: 500, width: '100%' }}>
      <AgGridReact<PlayerStatsRow>
        rowData={rowData}
        columnDefs={colDefs}
        defaultColDef={{
          flex: 1,
          minWidth: 100,
          resizable: true,
        }}
        pagination={true}
        paginationPageSize={10}
        paginationPageSizeSelector={[10, 25, 50]}
      />
    </div>
  );
};

export default DataGrid; 