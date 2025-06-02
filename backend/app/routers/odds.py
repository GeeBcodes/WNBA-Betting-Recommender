from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
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

@router.get("/game_odds/{game_odd_id}", response_model=odds_schema.GameOdd)
async def read_game_odd_endpoint(
    game_odd_id: uuid.UUID, 
    db: AsyncSession = Depends(get_db)
):
    db_game_odd = await crud.get_game_odd(db, game_odd_id=game_odd_id)
    if db_game_odd is None:
        raise HTTPException(status_code=404, detail="Game odd not found")
    return db_game_odd

@router.get("/game_odds/", response_model=List[odds_schema.GameOdd])
async def read_game_odds_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    game_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db)
):
    game_odds = await crud.get_game_odds(db, skip=skip, limit=limit, game_id=game_id)
    return game_odds

@router.get("/player_props/{player_prop_id}", response_model=odds_schema.PlayerProp)
async def read_player_prop_endpoint(
    player_prop_id: uuid.UUID, 
    db: AsyncSession = Depends(get_db)
):
    db_player_prop = await crud.get_player_prop(db, player_prop_id=player_prop_id)
    if db_player_prop is None:
        raise HTTPException(status_code=404, detail="Player prop not found")
    return db_player_prop

@router.get("/player_props/", response_model=List[odds_schema.PlayerProp])
async def read_player_props_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    game_id: Optional[uuid.UUID] = None, 
    player_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db)
):
    player_props = await crud.get_player_props(db, skip=skip, limit=limit, game_id=game_id, player_id=player_id)
    return player_props

# As noted in crud/odds.py, create/update/delete for odds are not implemented
# as they are handled by the scraper. If these were to be added, they would follow
# a similar pattern to other routers, e.g.:
#
# @router.post("/game_odds/", response_model=odds_schema.GameOdd, status_code=status.HTTP_201_CREATED)
# async def create_game_odd_endpoint(
#     game_odd: odds_schema.GameOddCreate, 
#     db: AsyncSession = Depends(get_db)
# ):
#     # Logic to check for existing game_odd or other validation
#     return await crud.create_game_odd(db=db, game_odd=game_odd) # Assuming create_game_odd exists
#
# Similarly for player props and other operations (update, delete) 