from pydantic import BaseModel, ConfigDict, validator, field_validator
from typing import List, Optional, Any, Dict
import uuid
from datetime import datetime

# Details for each selection in a parlay
class ParlaySelectionDetail(BaseModel):
    prediction_id: str # Changed to str from uuid.UUID to match what parlay_builder might send as JSON
    player_prop_id: str # Changed to str from uuid.UUID
    player_name: str
    market_key: str
    game_id: str # Changed to str from uuid.UUID
    line_point: Optional[float] = None
    chosen_outcome: str # e.g., "Over", "Under"
    chosen_probability: float

class ParlayBase(BaseModel):
    selections: List[ParlaySelectionDetail] # Store detailed selections
    combined_probability: Optional[float] = None
    total_odds: Optional[float] = None # Assuming decimal odds for now

    model_config = ConfigDict(
        protected_namespaces=() # Example, if needed later
    )

class ParlayCreate(ParlayBase):
    pass

class Parlay(ParlayBase):
    id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=() 
    ) 