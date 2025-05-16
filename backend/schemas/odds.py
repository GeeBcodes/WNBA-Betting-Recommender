from pydantic import BaseModel, ConfigDict
from datetime import datetime

class GameOddBase(BaseModel):
    home_team: str
    away_team: str
    home_team_odds: float | None = None
    away_team_odds: float | None = None
    spread: float | None = None
    over_under: float | None = None
    source: str
    last_updated: datetime | None = None

class GameOddCreate(GameOddBase):
    pass

class GameOdd(GameOddBase):
    game_id: str # Can be a unique string identifying the game

    model_config = ConfigDict(from_attributes=True)

class PlayerPropOddBase(BaseModel):
    player_name: str
    stat_type: str 
    line: float
    over_odds: int
    under_odds: int
    source: str
    last_updated: datetime | None = None

class PlayerPropOddCreate(PlayerPropOddBase):
    pass

class PlayerPropOdd(PlayerPropOddBase):
    prop_id: int # Assuming a unique ID for each prop bet
    player_id: int # Link to the player

    model_config = ConfigDict(from_attributes=True) 