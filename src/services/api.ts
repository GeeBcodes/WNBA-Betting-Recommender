import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000', // Your FastAPI backend URL
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- START: New/Updated Interfaces for Stats ---
// Matches backend.schemas.player.Player
export interface Player {
  id: string; // uuid.UUID
  player_name: string;
  team_name?: string | null;
  player_api_id?: string | null;
}

// Matches backend.schemas.game.Game
export interface Game {
  id: string; // uuid.UUID
  external_id?: string | null;
  home_team: string;
  away_team: string;
  game_datetime?: string | null; // datetime from backend
}

// Matches backend.schemas.player_stats.PlayerStatRead
export interface PlayerStatFull { // Renaming from PlayerStat to avoid confusion
  id: string; // uuid.UUID (for the stat record itself)
  player_id: string; // uuid.UUID (FK)
  game_id: string; // uuid.UUID (FK)
  game_date?: string | null; // date
  points?: number | null;
  rebounds?: number | null;
  assists?: number | null;
  steals?: number | null;
  blocks?: number | null;
  turnovers?: number | null;
  minutes_played?: number | null;
  field_goals_made?: number | null;
  field_goals_attempted?: number | null;
  three_pointers_made?: number | null;
  three_pointers_attempted?: number | null;
  free_throws_made?: number | null;
  free_throws_attempted?: number | null;
  plus_minus?: number | null;

  player: Player; // Nested Player object
  game: Game;     // Nested Game object
}
// --- END: New/Updated Interfaces for Stats ---

// --- Odds Related Interfaces (from backend/schemas/odds.py) ---
export interface Bookmaker {
  id: string;
  key: string;
  title: string;
}

export interface Market {
  id: string;
  key: string;
  description?: string | null;
}

export interface PlayerProp {
  id: string; // This is player_prop_id in Prediction
  game_id: string;
  player_id?: string | null;
  bookmaker_id: string;
  market_id: string;
  player_name_api?: string | null;
  last_update_api?: string | null; // datetime
  outcomes?: Array<{ name: string; price: number; point?: number }> | null; // Simplified, adjust if more complex

  // Expanded details
  bookmaker?: Bookmaker | null;
  market?: Market | null;
  player?: Player | null;
  game?: Game | null;
}
// --- End Odds Related Interfaces ---

export interface Prediction {
  id: string; 
  // player_prop_odd_id: string; // This will be replaced by the nested player_prop object
  // model_version_id: string; // To be removed from display
  predicted_over_probability: number | null; 
  predicted_under_probability: number | null; 
  // prediction_datetime: string; // To be removed from display

  player_prop?: PlayerProp | null; // Nested PlayerProp details
}

// Updated to match backend schemas/parlay.py ParlayCreate & ParlaySelectionDetail
export interface ParlaySelectionDetailPayload {
  prediction_id: string;
  player_prop_id: string;
  player_name: string;
  market_key: string;
  game_id: string;
  line_point: number | null;
  chosen_outcome: 'over' | 'under';
  chosen_probability: number;
}

export interface ParlayData {
  selections: ParlaySelectionDetailPayload[];
  combined_probability?: number | null;
  total_odds?: number | null; 
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
    // Explicitly request a larger limit
    const response = await apiClient.get<Prediction[]>('/predictions?limit=500'); // Max limit as per backend
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

// --- New API functions for expanded backend integration ---

// Model Versions
export interface ModelVersion {
  id: string;
  version_name: string;
  description?: string;
  trained_at: string;
}

export const getModelVersions = async (): Promise<ModelVersion[]> => {
  try {
    const response = await apiClient.get<ModelVersion[]>("/model_versions/");
    return response.data;
  } catch (error) {
    console.error("Error fetching model versions:", error);
    throw error;
  }
};

// Parlays
export interface Parlay {
  id: string;
  selections: any[];
  combined_probability?: number;
  total_odds?: number;
  created_at: string;
}

export const getParlays = async (): Promise<Parlay[]> => {
  try {
    const response = await apiClient.get<Parlay[]>("/parlays");
    return response.data;
  } catch (error) {
    console.error("Error fetching parlays:", error);
    throw error;
  }
};

// Player Stats
/* Commenting out old PlayerStat interface
export interface PlayerStat {
  player_id: string | number;
  player_name: string;
  team_name: string;
  points?: number;
  rebounds?: number;
  assists?: number;
  steals?: number;
  blocks?: number;
  turnovers?: number;
  minutes_played?: number;
  game_date?: string;
}
*/

export const getPlayerStats = async (): Promise<PlayerStatFull[]> => { // Updated return type
  try {
    // Request a larger number of records, e.g., 10000. Adjust as needed.
    const response = await apiClient.get<PlayerStatFull[]>("/api/stats?limit=10000"); // Updated generic type
    return response.data;
  } catch (error) {
    console.error("Error fetching player stats:", error);
    throw error;
  }
};

// Odds
export interface GameOdd {
  game_id: string;
  home_team: string;
  away_team: string;
  home_team_odds?: number;
  away_team_odds?: number;
  spread?: number;
  over_under?: number;
  source: string;
  last_updated?: string;
}

export interface PlayerPropOdd {
  prop_id: number;
  player_id: number;
  player_name: string;
  stat_type: string;
  line: number;
  over_odds: number;
  under_odds: number;
  source: string;
  last_updated?: string;
}

export const getGameOdds = async (): Promise<GameOdd[]> => {
  try {
    const response = await apiClient.get<GameOdd[]>("/api/odds/games");
    return response.data;
  } catch (error) {
    console.error("Error fetching game odds:", error);
    throw error;
  }
};

export const getPlayerPropOdds = async (playerId: number): Promise<PlayerPropOdd[]> => {
  try {
    const response = await apiClient.get<PlayerPropOdd[]>(`/api/odds/props/player/${playerId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching player prop odds for player ${playerId}:`, error);
    throw error;
  }
};

export default apiClient;