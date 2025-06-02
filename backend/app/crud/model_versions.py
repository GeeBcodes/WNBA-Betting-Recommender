import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session
import uuid
from typing import List, Optional

from backend.db import models
from backend.schemas import model_version as model_version_schema

# CRUD for ModelVersion
async def create_model_version(db: AsyncSession, model_version: model_version_schema.ModelVersionCreate) -> models.ModelVersion:
    db_model_version = models.ModelVersion(
        version_name=model_version.version_name,
        description=model_version.description,
        model_path=model_version.model_path,
        metrics=model_version.metrics
        # trained_at is default in model
    )
    db.add(db_model_version)
    await db.commit()
    await db.refresh(db_model_version)
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

async def get_model_versions(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.ModelVersion]:
    # TODO: Consider adding joinedload for predictions if needed in response
    stmt = select(models.ModelVersion).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all() 