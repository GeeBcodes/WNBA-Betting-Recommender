from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from .odds import PlayerPropRead # Added import for PlayerPropRead
# from .model_version import ModelVersion # Example for related schema, if needed for response

class PredictionBase(BaseModel):
    player_prop_id: uuid.UUID
    model_version_id: Optional[uuid.UUID] = None # Make optional as it's being deprecated
    model_version_id_reg: Optional[uuid.UUID] = None
    model_version_id_clf: Optional[uuid.UUID] = None
    
    predicted_over_probability: Optional[float] = None
    predicted_under_probability: Optional[float] = None
    predicted_value: Optional[float] = None
    # predicted_line: Optional[float] = None

    # New fields for actual outcomes
    actual_value: Optional[float] = None
    outcome: Optional[str] = None
    outcome_processed_at: Optional[datetime] = None

    # Fields for ICP Regression Output
    predicted_value_interval_lower: Optional[float] = None
    predicted_value_interval_upper: Optional[float] = None
    conformal_confidence_level_regr: Optional[float] = None

    # Field to store evaluations against multiple lines
    line_evaluations: Optional[List[Dict[str, Any]]] = None

    # Fields for ICP Classification Output
    prediction_set: Optional[List[str]] = None # e.g., ["Over"], ["Under"], ["Over", "Under"]
    over_p_value_calibrated: Optional[float] = None
    under_p_value_calibrated: Optional[float] = None
    conformal_confidence_level_clf: Optional[float] = None

    model_config = ConfigDict(
        protected_namespaces=(),
    )

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: uuid.UUID
    prediction_datetime: datetime
    player_prop: Optional[PlayerPropRead] = None
    # model_version: Optional[ModelVersion] = None # Example for related schema, if needed for response

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
    )

class PredictionRead(PredictionBase):
    id: uuid.UUID
    prediction_datetime: datetime
    player_prop: Optional[PlayerPropRead] = None # Fully nested PlayerProp object

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
    )

# ... rest of the file if any ... 