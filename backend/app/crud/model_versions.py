import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
import uuid
from typing import List, Optional

from backend.db import models
from backend.schemas import model_version as model_version_schema

logger = logging.getLogger(__name__)

# CRUD for ModelVersion
async def create_model_version(db: AsyncSession, model_version: model_version_schema.ModelVersionCreate) -> models.ModelVersion:
    """
    Creates a new model version in the database.
    Does NOT commit the session, this should be handled by the calling function's transaction block.
    """
    db_model_version = models.ModelVersion(
        model_name=model_version.model_name,
        feature_names=model_version.feature_names,
        version_name=model_version.version_name,
        description=model_version.description,
        model_path=model_version.model_path,
        metrics=model_version.metrics,
        parameters=model_version.parameters,
        model_type=model_version.model_type,
        nonconformity_scores_path=model_version.nonconformity_scores_path,
        nonconformity_scores_clf_path=model_version.nonconformity_scores_clf_path
    )
    db.add(db_model_version)
    await db.flush()
    await db.commit()
    
    logger.info(f"Successfully saved model version '{model_version.version_name}' to the database.")
    
    return db_model_version

async def get_model_version(db: AsyncSession, model_version_id: uuid.UUID) -> Optional[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    stmt = select(models.ModelVersion).filter(models.ModelVersion.id == model_version_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_model_version_by_name(db: AsyncSession, version_name: str) -> Optional[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    stmt = select(models.ModelVersion).filter(models.ModelVersion.version_name == version_name)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_latest_model_version_by_prefix(db: AsyncSession, prefix: str) -> Optional[models.ModelVersion]:
    """
    Fetches the most recent model version from the database that has a version_name
    starting with the given prefix.
    """
    stmt = (
        select(models.ModelVersion)
        .filter(models.ModelVersion.version_name.like(f"{prefix}%"))
        .order_by(models.ModelVersion.trained_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_model_versions(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    stmt = select(models.ModelVersion).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all() 