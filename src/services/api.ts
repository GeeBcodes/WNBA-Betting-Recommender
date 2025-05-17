import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000', // Your FastAPI backend URL
  headers: {
    'Content-Type': 'application/json',
  },
});

// Placeholder types - we will refine these later based on backend schemas
// TEMPORARILY ADJUSTED FOR CURRENT BACKEND SCHEMA - NEEDS REFINEMENT
export interface Prediction {
  id: string; // Changed from number to string to accommodate UUID from backend for now
  player_prop_odd_id: string; // Added based on backend schema
  model_version_id: string; // Added based on backend schema
  predicted_over_probability: number | null; // Aligned with backend schema (Optional[float])
  predicted_under_probability: number | null; // Aligned with backend schema (Optional[float])
  prediction_datetime: string; // Added, assuming datetime becomes string

  // Fields that DataGrid was expecting, now missing or needing mapping:
  // game_id: string;
  // player_id: string;
  // market: string; 
  // line: number;
}

export interface ParlayData {
  name: string;
  legs: Array<{
    prediction_id: number;
    type: 'over' | 'under';
  }>;
  // Add other relevant fields for creating a parlay
}

export const getGames = async () => {
  try {
    const response = await apiClient.get('/api/games');
    return response.data;
  } catch (error) {
    console.error('Error fetching games:', error);
    throw error;
  }
};

export const getPlayers = async () => {
  try {
    const response = await apiClient.get('/api/players');
    return response.data;
  } catch (error) {
    console.error('Error fetching players:', error);
    throw error;
  }
};

export const getOdds = async (gameId: string) => {
  try {
    const response = await apiClient.get(`/api/odds?game_id=${gameId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching odds for game ${gameId}:`, error);
    throw error;
  }
};

// New functions for Phase 4
export const getPredictions = async (): Promise<Prediction[]> => {
  try {
    const response = await apiClient.get<Prediction[]>('/predictions');
    return response.data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    // It's good practice to inform the user or log this more formally
    // For now, we'll re-throw to be handled by the calling component
    throw error;
  }
};

export const postParlay = async (parlayData: ParlayData): Promise<any> => { // Replace 'any' with a more specific Parlay response type later
  try {
    const response = await apiClient.post('/parlays', parlayData);
    return response.data;
  } catch (error) {
    console.error('Error posting parlay:', error);
    // Similar to getPredictions, handle error appropriately
    throw error;
  }
};

export default apiClient;