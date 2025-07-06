from pydantic import BaseModel
import uuid
from typing import Optional

class TeamBase(BaseModel):
    team_name: str
    api_team_id: str
    abbreviation: Optional[str] = None

class TeamCreate(TeamBase):
    pass

class Team(TeamBase):
    id: uuid.UUID

    class Config:
        from_attributes = True 