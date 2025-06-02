import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List, Optional

from backend.db import models
from backend.schemas import game as game_schema

# --- CRUD for Game ---
async def get_game(db: AsyncSession, game_id: uuid.UUID) -> Optional[models.Game]:
    stmt = select(models.Game).filter(models.Game.id == game_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_game_by_external_id(db: AsyncSession, external_id: str) -> Optional[models.Game]:
    # Assuming Game model has an 'external_id' field for IDs from data sources like sportsdataverse
    stmt = select(models.Game).filter(models.Game.external_id == external_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_games(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Game]:
    stmt = select(models.Game).order_by(models.Game.game_datetime.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_game(db: AsyncSession, game: game_schema.GameCreate) -> models.Game:
    # Ensure that all fields expected by models.Game are present in game_schema.GameCreate
    # or handle missing fields appropriately.
    db_game = models.Game(**game.model_dump())
    db.add(db_game)
    await db.commit()
    await db.refresh(db_game)
    return db_game 