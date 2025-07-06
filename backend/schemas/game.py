from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
import uuid

class GameBase(BaseModel):
    the_odds_api_game_id: Optional[str] = None
    home_team_id: Optional[uuid.UUID] = None
    away_team_id: Optional[uuid.UUID] = None
    game_datetime: datetime
    season: int
    status: str
    home_team_score: Optional[int] = None
    away_team_score: Optional[int] = None
    home_team_possessions: Optional[int] = None
    away_team_possessions: Optional[int] = None

class GameCreate(GameBase):
    pass

class Game(GameBase):
    id: uuid.UUID
    model_config = ConfigDict(from_attributes=True) 