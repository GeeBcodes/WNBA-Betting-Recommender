from fastapi import APIRouter
from typing import List
from backend.schemas.odds import GameOdd, PlayerPropOdd 
from datetime import datetime

router = APIRouter(
    prefix="/api/odds",
    tags=["Betting Odds"],
)

# Placeholder data for now
mock_game_odds_db = {
    "GAME123": GameOdd(game_id="GAME123", home_team="Las Vegas Aces", away_team="New York Liberty", home_team_odds=-150, away_team_odds=130, spread=-3.5, over_under=165.5, source="MockSportsbook", last_updated=datetime.now()),
}

mock_player_props_db = {
    1: PlayerPropOdd(prop_id=1, player_id=1, player_name="A'ja Wilson", stat_type="points", line=24.5, over_odds=-110, under_odds=-110, source="MockSportsbook", last_updated=datetime.now()),
}

@router.get("/games", response_model=List[GameOdd])
async def get_all_game_odds():
    """Retrieve all available game odds."""
    return list(mock_game_odds_db.values())

@router.get("/games/{game_id}", response_model=GameOdd)
async def get_game_odds_by_id(game_id: str):
    """Retrieve odds for a specific game by its ID."""
    if game_id in mock_game_odds_db:
        return mock_game_odds_db[game_id]
    return {"error": "Game odds not found"}

@router.get("/props/player/{player_id}", response_model=List[PlayerPropOdd])
async def get_player_prop_odds(player_id: int):
    """Retrieve all prop bets for a specific player."""
    # This is a simplified example
    results = [prop for prop in mock_player_props_db.values() if prop.player_id == player_id]
    return results 