from pydantic import BaseModel, Json
from typing import Optional, List, Any
import uuid
from datetime import datetime

class ParlayBase(BaseModel):
    selections: List[Any] # Could be list of prediction_ids (UUIDs) or more complex objects
    combined_probability: Optional[float] = None
    total_odds: Optional[float] = None
    # user_id: Optional[uuid.UUID] = None # If users are added

class ParlayCreate(ParlayBase):
    pass

class Parlay(ParlayBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        # orm_mode = True # Pydantic V1 way
        from_attributes = True # Pydantic V2 way 