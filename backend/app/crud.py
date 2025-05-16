from sqlalchemy.orm import Session
import uuid # For UUID type
from typing import List, Optional # For list and optional types

from backend.db import models
from backend.schemas import model_version as model_version_schema # Alias to avoid name collision
from backend.schemas import prediction as prediction_schema # Added for Prediction CRUD
from backend.schemas import parlay as parlay_schema # Added for Parlay CRUD

# CRUD for ModelVersion

def create_model_version(db: Session, model_version: model_version_schema.ModelVersionCreate) -> models.ModelVersion:
    db_model_version = models.ModelVersion(
        version_name=model_version.version_name,
        description=model_version.description
        # trained_at is default in model
    )
    db.add(db_model_version)
    db.commit()
    db.refresh(db_model_version)
    return db_model_version

def get_model_version(db: Session, model_version_id: uuid.UUID) -> Optional[models.ModelVersion]:
    return db.query(models.ModelVersion).filter(models.ModelVersion.id == model_version_id).first()

def get_model_version_by_name(db: Session, version_name: str) -> Optional[models.ModelVersion]:
    return db.query(models.ModelVersion).filter(models.ModelVersion.version_name == version_name).first()

def get_model_versions(db: Session, skip: int = 0, limit: int = 100) -> List[models.ModelVersion]:
    return db.query(models.ModelVersion).offset(skip).limit(limit).all()

# We might not need update/delete for model versions initially, or they might have specific logic
# (e.g., can only delete if no predictions are linked). For now, let's omit them.
# If needed, they would look like:
# def update_model_version(db: Session, model_version_id: uuid.UUID, model_version_update: model_version_schema.ModelVersionUpdate) -> Optional[models.ModelVersion]:
#     db_model_version = get_model_version(db, model_version_id)
#     if db_model_version:
#         update_data = model_version_update.model_dump(exclude_unset=True) # Pydantic v2
#         # update_data = model_version_update.dict(exclude_unset=True) # Pydantic v1
#         for key, value in update_data.items():
#             setattr(db_model_version, key, value)
#         db.commit()
#         db.refresh(db_model_version)
#     return db_model_version

# def delete_model_version(db: Session, model_version_id: uuid.UUID) -> Optional[models.ModelVersion]:
#     db_model_version = get_model_version(db, model_version_id)
#     if db_model_version:
#         db.delete(db_model_version)
#         db.commit()
#     return db_model_version

# --- CRUD for Prediction ---
def create_prediction(db: Session, prediction: prediction_schema.PredictionCreate) -> models.Prediction:
    db_prediction = models.Prediction(
        player_prop_odd_id=prediction.player_prop_odd_id,
        model_version_id=prediction.model_version_id,
        predicted_over_probability=prediction.predicted_over_probability,
        predicted_under_probability=prediction.predicted_under_probability
        # prediction_datetime is default in model
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction(db: Session, prediction_id: uuid.UUID) -> Optional[models.Prediction]:
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()

def get_predictions(db: Session, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    return db.query(models.Prediction).offset(skip).limit(limit).all()

def get_predictions_by_player_prop_odd(db: Session, player_prop_odd_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    return db.query(models.Prediction).filter(models.Prediction.player_prop_odd_id == player_prop_odd_id).offset(skip).limit(limit).all()

def get_predictions_by_model_version(db: Session, model_version_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    return db.query(models.Prediction).filter(models.Prediction.model_version_id == model_version_id).offset(skip).limit(limit).all()

# Update/Delete for Predictions might also have specific logic (e.g., soft delete, or preventing updates)
# For now, omitting update/delete for Predictions.

# --- CRUD for Parlay ---
def create_parlay(db: Session, parlay: parlay_schema.ParlayCreate) -> models.Parlay:
    db_parlay = models.Parlay(
        selections=parlay.selections, # This will be JSON
        combined_probability=parlay.combined_probability,
        total_odds=parlay.total_odds
        # created_at is default in model
        # user_id can be added later if users are implemented
    )
    db.add(db_parlay)
    db.commit()
    db.refresh(db_parlay)
    return db_parlay

def get_parlay(db: Session, parlay_id: uuid.UUID) -> Optional[models.Parlay]:
    return db.query(models.Parlay).filter(models.Parlay.id == parlay_id).first()

def get_parlays(db: Session, skip: int = 0, limit: int = 100) -> List[models.Parlay]: 
    return db.query(models.Parlay).offset(skip).limit(limit).all()

# Update for Parlay might involve recalculating odds/probability if selections change.
# Delete for Parlay is straightforward if it's a hard delete.
# Omitting update/delete for now for simplicity. 