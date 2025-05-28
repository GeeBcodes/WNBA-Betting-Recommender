from pydantic import BaseModel, ConfigDict
from datetime import date, datetime
from typing import Optional
import uuid

# Import schemas for nesting
from .player import Player
from .game import Game

class PlayerStatBase(BaseModel):
    player_id: uuid.UUID
    game_id: uuid.UUID

    game_date: Optional[date] = None
    points: Optional[float] = None
    rebounds: Optional[float] = None
    assists: Optional[float] = None
    steals: Optional[float] = None
    blocks: Optional[float] = None
    turnovers: Optional[float] = None
    minutes_played: Optional[float] = None
    field_goals_made: Optional[int] = None
    field_goals_attempted: Optional[int] = None
    three_pointers_made: Optional[int] = None
    three_pointers_attempted: Optional[int] = None
    free_throws_made: Optional[int] = None
    free_throws_attempted: Optional[int] = None
    plus_minus: Optional[int] = None

class PlayerStatCreate(PlayerStatBase):
    pass

class PlayerStat(PlayerStatBase):
    id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)

class PlayerStatRead(PlayerStat):
    player: Player
    game: Game

    # The model_config is inherited from PlayerStat, so no need to repeat it here. 