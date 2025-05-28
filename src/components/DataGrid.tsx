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
  // Updated to match the TEMPORARILY ADJUSTED Prediction interface fields
  const colDefs = useMemo<ColDef<Prediction>[]>(() => [
    {
      headerName: 'Select',
      checkboxSelection: true,
      headerCheckboxSelection: true, // Optional: for select/deselect all
      width: 100,
      pinned: 'left', // Optional: pin the select column
    },
    { field: "id", headerName: "Pred. ID", filter: true, sortable: true, minWidth: 250 },
    { field: "player_prop_odd_id", headerName: "Prop Odd ID", filter: true, sortable: true, minWidth: 250 },
    { field: "model_version_id", headerName: "Model Ver. ID", filter: true, sortable: true, minWidth: 250 },
    { field: "prediction_datetime", headerName: "Timestamp", filter: true, sortable: true, minWidth: 180 },
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
    // NOTE: Missing fields for user-friendly display (player name, game, market, line) will be addressed later
    // by enhancing backend response for GET /predictions.
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