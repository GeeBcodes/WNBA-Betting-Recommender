from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
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
async def read_player_stats_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    player_id: Optional[uuid.UUID] = None,
    game_id: Optional[uuid.UUID] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve player statistics with optional filtering.
    - **player_id**: Filter by player's database ID.
    - **game_id**: Filter by game's database ID.
    - **start_date**: Filter for stats on or after this date (YYYY-MM-DD).
    - **end_date**: Filter for stats on or before this date (YYYY-MM-DD).
    """
    player_stats = await crud.get_player_stats(
        db, 
        skip=skip, 
        limit=limit, 
        player_id=player_id, 
        game_id=game_id,
        start_date=start_date,
        end_date=end_date
    )
    return player_stats

@router.get("/{player_stat_id}", response_model=player_stat_schema.PlayerStatRead)
async def read_player_stat_endpoint(
    player_stat_id: uuid.UUID, 
    db: AsyncSession = Depends(get_db)
):
    db_player_stat = await crud.get_player_stat(db, player_stat_id=player_stat_id)
    if db_player_stat is None:
        raise HTTPException(status_code=404, detail="Player stat not found")
    return db_player_stat

# The /{player_id} endpoint is removed as its functionality is covered by GET /?player_id=...
# If a specific endpoint for a single PlayerStat by its own ID is needed, it can be added:
# @router.get("/{stat_id}", response_model=player_stat_schema.PlayerStatRead)
# async def read_single_player_stat(stat_id: uuid.UUID, db: Session = Depends(get_db)):
#     db_stat = crud.get_player_stat(db, player_stat_id=stat_id) # Assumes get_player_stat is modified for joinedload
#     if db_stat is None:
#         raise HTTPException(status_code=404, detail="Player stat not found")
#     return db_stat 

# As player stats are typically derived from game data and not directly created/updated/deleted via API,
# POST, PUT, DELETE endpoints are omitted for now.
# If direct manipulation is needed, they would be added similarly to other routers.
# For example:
#
# @router.post("/", response_model=player_stat_schema.PlayerStat, status_code=status.HTTP_201_CREATED)
# async def create_player_stat_endpoint(
#     player_stat: player_stat_schema.PlayerStatCreate, 
#     db: AsyncSession = Depends(get_db)
# ):
#     # Add any necessary validation or pre-processing
#     return await crud.create_player_stat(db=db, player_stat=player_stat) # Assuming create_player_stat exists in crud 