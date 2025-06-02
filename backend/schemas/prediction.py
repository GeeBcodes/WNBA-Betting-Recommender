from pydantic import BaseModel, ConfigDict
from typing import Optional
import uuid
from datetime import datetime
from .odds import PlayerPropRead # Added import for PlayerPropRead
# from .model_version import ModelVersion # Example for related schema, if needed for response

class PredictionBase(BaseModel):
    player_prop_id: uuid.UUID
    model_version_id: uuid.UUID
    predicted_over_probability: Optional[float] = None
    predicted_under_probability: Optional[float] = None
    predicted_value: Optional[float] = None
    # predicted_line: Optional[float] = None

    # New fields for actual outcomes
    actual_value: Optional[float] = None
    outcome: Optional[str] = None
    outcome_processed_at: Optional[datetime] = None

    # Add model_config here to cover all schemas using model_version_id
    model_config = ConfigDict(
        protected_namespaces=()
    )

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: uuid.UUID
    prediction_datetime: datetime
    player_prop: Optional[PlayerPropRead] = None
    # model_version: Optional[ModelVersion] = None # Example for related schema, if needed for response

    # Use ConfigDict for Pydantic V2 model configuration
    # This config is now inherited from PredictionBase, but from_attributes might be specific to Prediction
    model_config = ConfigDict(
        from_attributes=True, # Keep this if only Prediction schema is mapped from ORM
        protected_namespaces=() # This is inherited, but being explicit doesn't hurt
    )

# ... rest of the file if any ... 