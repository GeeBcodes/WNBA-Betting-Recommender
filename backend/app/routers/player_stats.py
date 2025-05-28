from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session
import uuid
from datetime import date

from schemas import player_stats as player_stat_schema
from ..dependencies import get_db
from .. import crud

router = APIRouter(
    prefix="/api/stats",
    tags=["Player Stats"],
)

@router.get("/", response_model=List[player_stat_schema.PlayerStatRead])
async def read_player_stats(
    skip: int = 0, 
    limit: int = 100, 
    player_id: Optional[uuid.UUID] = None,
    game_id: Optional[uuid.UUID] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve player statistics with optional filtering.
    - **player_id**: Filter by player's database ID.
    - **game_id**: Filter by game's database ID.
    - **start_date**: Filter for stats on or after this date (YYYY-MM-DD).
    - **end_date**: Filter for stats on or before this date (YYYY-MM-DD).
    """
    stats = crud.get_player_stats(
        db, skip=skip, limit=limit, 
        player_id=player_id, game_id=game_id,
        start_date=start_date, end_date=end_date
    )
    if not stats:
        # Return empty list if no stats match, or you could raise HTTPException(status_code=404, detail="No stats found")
        # For a list endpoint, returning an empty list is often standard.
        return []
    return stats

# The /{player_id} endpoint is removed as its functionality is covered by GET /?player_id=...
# If a specific endpoint for a single PlayerStat by its own ID is needed, it can be added:
# @router.get("/{stat_id}", response_model=player_stat_schema.PlayerStatRead)
# async def read_single_player_stat(stat_id: uuid.UUID, db: Session = Depends(get_db)):
#     db_stat = crud.get_player_stat(db, player_stat_id=stat_id) # Assumes get_player_stat is modified for joinedload
#     if db_stat is None:
#         raise HTTPException(status_code=404, detail="Player stat not found")
#     return db_stat 