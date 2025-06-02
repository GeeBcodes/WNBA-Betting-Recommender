from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from .player import Player # Import Player schema
from .game import Game # Import Game schema

# --- Sport Schemas ---
class SportBase(BaseModel):
    key: str
    group_name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    active: bool = True
    has_outrights: bool = False

class SportCreate(SportBase):
    pass

class Sport(SportBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# --- Bookmaker Schemas ---
class BookmakerBase(BaseModel):
    key: str
    title: str

class BookmakerCreate(BookmakerBase):
    pass

class Bookmaker(BookmakerBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# --- Market Schemas ---
class MarketBase(BaseModel):
    key: str
    description: Optional[str] = None

class MarketCreate(MarketBase):
    pass

class Market(MarketBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)


# --- GameOdd Schemas ---
class GameOddBase(BaseModel):
    game_id: uuid.UUID
    bookmaker_id: uuid.UUID
    market_id: uuid.UUID
    last_update_api: Optional[datetime] = None
    outcomes: Optional[List[Dict[str, Any]]] = None

class GameOddCreate(GameOddBase):
    pass

class GameOdd(GameOddBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)

# Read schema with related objects if needed later
class GameOddRead(GameOdd):
    bookmaker: Optional[Bookmaker] = None 
    market: Optional[Market] = None
    # game: Optional[Game] # Assuming you have a Game schema in game.py


# --- PlayerProp Schemas ---
class PlayerPropBase(BaseModel):
    game_id: uuid.UUID
    player_id: Optional[uuid.UUID] = None
    bookmaker_id: uuid.UUID
    market_id: uuid.UUID
    player_name_api: Optional[str] = None
    last_update_api: Optional[datetime] = None
    outcomes: Optional[List[Dict[str, Any]]] = None

class PlayerPropCreate(PlayerPropBase):
    pass

class PlayerProp(PlayerPropBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)

# Read schema with related objects
class PlayerPropRead(PlayerProp):
    bookmaker: Optional[Bookmaker] = None
    market: Optional[Market] = None
    player: Optional[Player] = None # Uncommented and using imported Player schema
    game: Optional[Game] = None # Uncommented and using imported Game schema


# Update __all__ in backend/schemas/__init__.py if you create it 