from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

class ModelVersionBase(BaseModel):
    version_name: str
    description: Optional[str] = None
    # parameters: Optional[dict] = None # If you decide to store parameters
    # accuracy: Optional[float] = None   # If you decide to store accuracy

class ModelVersionCreate(ModelVersionBase):
    pass

class ModelVersion(ModelVersionBase):
    id: uuid.UUID
    trained_at: datetime
    # predictions: List['Prediction'] = [] # Avoid circular dependency if Prediction schema also refers to this

    class Config:
        # orm_mode = True # Pydantic V1 way, for Pydantic V2 use from_attributes=True
        from_attributes = True # Pydantic V2 way 