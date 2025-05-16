from pydantic import BaseModel, ConfigDict
from typing import Optional
import uuid
from datetime import datetime
# from .player_prop_odd import PlayerPropOdd # Example for related schema, if needed for response
# from .model_version import ModelVersion # Example for related schema, if needed for response

class PredictionBase(BaseModel):
    player_prop_odd_id: uuid.UUID
    model_version_id: uuid.UUID
    predicted_over_probability: Optional[float] = None
    predicted_under_probability: Optional[float] = None
    # predicted_line: Optional[float] = None

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: uuid.UUID
    prediction_datetime: datetime
    # player_prop_odd: Optional[PlayerPropOdd] = None # For richer response model
    # model_version: Optional[ModelVersion] = None # For richer response model

    # Use ConfigDict for Pydantic V2 model configuration
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )

# ... rest of the file if any ... 