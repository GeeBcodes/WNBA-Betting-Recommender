from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import uuid

# Base Schema for a Parlay Leg
class ParlayLegBase(BaseModel):
    parlay_id: uuid.UUID
    prediction_id: uuid.UUID
    model_version_id: uuid.UUID
    leg_number: int
    odds: float
    bet_amount: float
    status: str = Field(default='pending')


class ParlayLegCreate(ParlayLegBase):
    pass


class ParlayLeg(ParlayLegBase):
    id: uuid.UUID
    created_at: datetime
    
    class Config:
        from_attributes = True
        protected_namespaces = ()


# Base Schema for a Parlay
class ParlayBase(BaseModel):
    user_id: uuid.UUID # Assuming a user model exists
    total_odds: float
    total_bet_amount: float
    potential_payout: float
    status: str = Field(default='pending')


class ParlayCreate(ParlayBase):
    pass


class Parlay(ParlayBase):
    id: uuid.UUID
    created_at: datetime
    legs: List['ParlayLeg'] = []
    
    class Config:
        from_attributes = True 