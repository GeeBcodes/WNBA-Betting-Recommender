from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
import uuid
from datetime import datetime

class ModelVersionBase(BaseModel):
    version_name: str
    description: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None
    # parameters: Optional[dict] = None # If you decide to store parameters
    # accuracy: Optional[float] = None   # If you decide to store accuracy

    # Add model_config here to cover all schemas using these fields
    model_config = ConfigDict(
        protected_namespaces=()
    )

class ModelVersionCreate(ModelVersionBase):
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None

class ModelVersion(ModelVersionBase):
    id: uuid.UUID
    trained_at: datetime
    # predictions: List['Prediction'] = [] # Avoid circular dependency if Prediction schema also refers to this

    # ConfigDict is inherited from ModelVersionBase, but from_attributes might be specific
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=() # Ensure it's here too if not fully inherited or overridden
    ) 