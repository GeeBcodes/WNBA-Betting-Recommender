from sqlalchemy.orm import Session
import uuid
from typing import List, Optional

from db import models
from schemas import model_version as model_version_schema

# CRUD for ModelVersion
def create_model_version(db: Session, model_version: model_version_schema.ModelVersionCreate) -> models.ModelVersion:
    db_model_version = models.ModelVersion(
        version_name=model_version.version_name,
        description=model_version.description,
        model_path=model_version.model_path,
        metrics=model_version.metrics
        # trained_at is default in model
    )
    db.add(db_model_version)
    db.commit()
    db.refresh(db_model_version)
    return db_model_version

def get_model_version(db: Session, model_version_id: uuid.UUID) -> Optional[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    return db.query(models.ModelVersion).filter(models.ModelVersion.id == model_version_id).first()

def get_model_version_by_name(db: Session, version_name: str) -> Optional[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    return db.query(models.ModelVersion).filter(models.ModelVersion.version_name == version_name).first()

def get_model_versions(db: Session, skip: int = 0, limit: int = 100) -> List[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    return db.query(models.ModelVersion).offset(skip).limit(limit).all() 