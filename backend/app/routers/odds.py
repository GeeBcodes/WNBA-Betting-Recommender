from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session
import uuid # Added import

from schemas import odds as odds_schema # Use alias for Pydantic schemas
from ..dependencies import get_db
from .. import crud # Import the main crud module

router = APIRouter(
    prefix="/api/odds",
    tags=["Betting Odds"],
)

@router.get("/games", response_model=List[odds_schema.GameOddRead])
async def read_all_game_odds(
    skip: int = 0, 
    limit: int = 100, 
    game_id: Optional[uuid.UUID] = None,
    db: Session = Depends(get_db)
):
    """Retrieve all available game odds, with optional filtering by game_id."""
    game_odds_list = crud.get_game_odds(db, skip=skip, limit=limit, game_id=game_id)
    if not game_odds_list:
        return []
    return game_odds_list

@router.get("/games/{game_odd_id}", response_model=odds_schema.GameOddRead)
async def read_game_odd_by_id(game_odd_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve specific game odds by its primary ID."""
    db_game_odd = crud.get_game_odd(db, game_odd_id=game_odd_id)
    if db_game_odd is None:
        raise HTTPException(status_code=404, detail="Game odds not found")
    return db_game_odd

@router.get("/props", response_model=List[odds_schema.PlayerPropRead])
async def read_all_player_props(
    skip: int = 0, 
    limit: int = 100, 
    game_id: Optional[uuid.UUID] = None,
    player_id: Optional[uuid.UUID] = None,
    db: Session = Depends(get_db)
):
    """Retrieve all player props, with optional filtering by game_id and player_id."""
    player_props_list = crud.get_player_props(
        db, skip=skip, limit=limit, game_id=game_id, player_id=player_id
    )
    if not player_props_list:
        return []
    return player_props_list

@router.get("/props/{player_prop_id}", response_model=odds_schema.PlayerPropRead)
async def read_player_prop_by_id(player_prop_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve a specific player prop by its primary ID."""
    db_player_prop = crud.get_player_prop(db, player_prop_id=player_prop_id)
    if db_player_prop is None:
        raise HTTPException(status_code=404, detail="Player prop not found")
    return db_player_prop

# Example of a more specific endpoint, similar to what was there before:
@router.get("/props/player/{player_id}", response_model=List[odds_schema.PlayerPropRead])
async def read_player_props_for_player(
    player_id: uuid.UUID, 
    game_id: Optional[uuid.UUID] = None, # Optional filter by game
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Retrieve all prop bets for a specific player, optionally filtered by game."""
    player_props_list = crud.get_player_props(
        db, player_id=player_id, game_id=game_id, skip=skip, limit=limit
    )
    if not player_props_list:
        # Depending on desired behavior, could raise 404 if no props found for a player
        # or return empty list if that's acceptable.
        return [] 
    return player_props_list 