import React, { useMemo, useRef, useCallback } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, SelectionChangedEvent } from 'ag-grid-community';
import { Prediction } from '../services/api'; // Import the Prediction interface

// Define props for the DataGrid component
interface DataGridProps {
  predictions: Prediction[];
  onSelectionChanged: (selectedPredictions: Prediction[]) => void; // Callback for selection changes
}

const DataGrid: React.FC<DataGridProps> = ({ predictions, onSelectionChanged }) => {
  const gridRef = useRef<AgGridReact<Prediction>>(null); // Ref for accessing Grid API

  // Column Definitions: Defines & controls grid columns.
  const colDefs = useMemo<ColDef<Prediction>[]>(() => [
    {
      headerName: 'Select',
      checkboxSelection: true,
      headerCheckboxSelection: true,
      width: 100,
      pinned: 'left',
    },
    { 
      headerName: "Player", 
      valueGetter: p => p.data?.player_prop?.player?.player_name || p.data?.player_prop?.player_name_api || 'N/A', 
      filter: true, 
      sortable: true, 
      minWidth: 150 
    },
    { 
      headerName: "Game", 
      valueGetter: p => {
        const game = p.data?.player_prop?.game;
        const awayTeam = game?.away_team || 'N/A';
        const homeTeam = game?.home_team || 'N/A';
        if (awayTeam === 'N/A' && homeTeam === 'N/A') return 'N/A';
        return `${awayTeam} @ ${homeTeam}`;
      }, 
      filter: true, 
      sortable: true, 
      minWidth: 200 
    },
    { 
      headerName: "Market", 
      valueGetter: p => p.data?.player_prop?.market?.description || p.data?.player_prop?.market?.key || 'N/A', 
      filter: true, 
      sortable: true, 
      minWidth: 180 
    },
    {
      headerName: "Line",
      valueGetter: p => {
        const outcomes = p.data?.player_prop?.outcomes;
        // Assuming the first outcome contains the relevant 'point' for the line
        // This might need adjustment based on the actual structure of 'outcomes'
        if (outcomes && outcomes.length > 0 && typeof outcomes[0].point !== 'undefined') {
          return outcomes[0].point;
        }
        return 'N/A';
      },
      sortable: true,
      filter: 'agNumberColumnFilter',
      minWidth: 100
    },
    {
      field: "predicted_over_probability",
      headerName: "Over Prob.",
      sortable: true,
      valueFormatter: p => (typeof p.value === 'number' ? (p.value * 100).toFixed(1) + '%' : 'N/A'),
      cellDataType: 'number',
      minWidth: 120
    },
    {
      field: "predicted_under_probability",
      headerName: "Under Prob.",
      sortable: true,
      valueFormatter: p => (typeof p.value === 'number' ? (p.value * 100).toFixed(1) + '%' : 'N/A'),
      cellDataType: 'number',
      minWidth: 120
    },
    // Removed: player_prop_odd_id, model_version_id, prediction_datetime
  ], []);

  // Callback for when grid selection changes
  const handleSelectionChanged = useCallback((event: SelectionChangedEvent<Prediction>) => {
    if (event.api) {
      onSelectionChanged(event.api.getSelectedRows());
    }
  }, [onSelectionChanged]);

  return (
    <div className="ag-theme-quartz" style={{ height: 500, width: '100%' }}>
      <AgGridReact<Prediction>
        ref={gridRef} // Assign ref
        rowData={predictions}
        columnDefs={colDefs}
        rowSelection='multiple' // Enable multiple row selection
        suppressRowClickSelection={true} // Rows are selected via checkbox only
        onSelectionChanged={handleSelectionChanged} // Handle selection changes
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