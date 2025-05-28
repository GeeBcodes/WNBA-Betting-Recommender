from pydantic import BaseModel, ConfigDict
from typing import Optional
import uuid

class PlayerBase(BaseModel):
    player_name: str
    team_name: Optional[str] = None
    player_api_id: Optional[str] = None # Added based on scraper usage

class PlayerCreate(PlayerBase):
    pass

class Player(PlayerBase):
    id: uuid.UUID
    model_config = ConfigDict(from_attributes=True) 