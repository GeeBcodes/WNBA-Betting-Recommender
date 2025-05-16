from fastapi import APIRouter
from typing import List
from backend.schemas.player_stats import PlayerStat

router = APIRouter(
    prefix="/api/stats",
    tags=["Player Stats"],
)

# Placeholder data for now
mock_player_stats_db = {
    1: PlayerStat(player_id=1, player_name="A'ja Wilson", team_name="Las Vegas Aces", points=25.0, rebounds=10.0, game_date="2023-10-10"),
    2: PlayerStat(player_id=2, player_name="Breanna Stewart", team_name="New York Liberty", points=22.5, assists=5.0, game_date="2023-10-10"),
}

@router.get("/", response_model=List[PlayerStat])
async def get_all_player_stats():
    """Retrieve all player statistics."""
    return list(mock_player_stats_db.values())

@router.get("/{player_id}", response_model=PlayerStat)
async def get_player_stats_by_id(player_id: int):
    """Retrieve statistics for a specific player by their ID."""
    
    if player_id in mock_player_stats_db:
        return mock_player_stats_db[player_id]
    return {"error": "Player not found"} # Or raise HTTPException 