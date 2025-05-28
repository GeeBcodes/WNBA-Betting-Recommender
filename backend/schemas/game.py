from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
import uuid

class GameBase(BaseModel):
    external_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_datetime: Optional[datetime] = None

class GameCreate(GameBase):
    pass

class Game(GameBase):
    id: uuid.UUID
    model_config = ConfigDict(from_attributes=True) 