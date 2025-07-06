from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

class ModelVersionBase(BaseModel):
    version_name: str
    model_type: str
    description: Optional[str] = None
    model_path: Optional[str] = None
    pipeline_path: Optional[str] = None
    metrics: Optional[Dict] = None
    parameters: Optional[Dict[str, Any]] = None
    nonconformity_scores_path: Optional[str] = None # For regression
    nonconformity_scores_clf_path: Optional[str] = None # For classification
    model_name: str
    target_stat: str
    seasons: List[int]
    feature_names: List[str]
    # accuracy: Optional[float] = None   # If you decide to store accuracy

    # Add model_config here to cover all schemas using these fields
    model_config = ConfigDict(
        protected_namespaces=()
    )

class ModelVersionCreate(ModelVersionBase):
    model_uuid: str
    version: int
    training_date: datetime
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None
    parameters: Optional[Dict[str, Any]] = None
    nonconformity_scores_path: Optional[str] = None # For regression
    nonconformity_scores_clf_path: Optional[str] = None # For classification

class ModelVersion(ModelVersionBase):
    id: uuid.UUID
    trained_at: datetime
    # predictions: List['Prediction'] = [] # Avoid circular dependency if Prediction schema also refers to this
    nonconformity_scores_path: Optional[str] = None # For regression
    nonconformity_scores_clf_path: Optional[str] = None # For classification

    # ConfigDict is inherited from ModelVersionBase, but from_attributes might be specific
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=() # Ensure it's here too if not fully inherited or overridden
    ) 