import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List, Optional

from backend.db import models
from backend.schemas import player as player_schema

# --- CRUD for Player ---
async def get_player(db: AsyncSession, player_id: uuid.UUID) -> Optional[models.Player]:
    stmt = select(models.Player).filter(models.Player.id == player_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_player_by_api_id(db: AsyncSession, player_api_id: str) -> Optional[models.Player]: # This might be wnba_player_id
    # Assuming player_api_id corresponds to wnba_player_id or a similar unique external ID
    # If models.Player has a specific field like `external_api_id` or `wnba_player_id`, use that.
    # For now, using a hypothetical `player_api_id` field.
    # Replace `models.Player.player_api_id` with the actual field name if different.
    # stmt = select(models.Player).filter(models.Player.player_api_id == player_api_id)
    # result = await db.execute(stmt)
    # return result.scalars().first()
    # Based on current models.Player, there is no player_api_id. Only player_name is unique.
    # This function might need revision based on how players are uniquely identified from APIs.
    # For now, let's assume player_name is used if no other API ID field exists.
    stmt = select(models.Player).filter(models.Player.player_name == player_api_id) # Fallback to name if no API ID
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_players(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Player]:
    stmt = select(models.Player).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_player(db: AsyncSession, player: player_schema.PlayerCreate) -> models.Player:
    # Ensure that all fields expected by models.Player are present in player_schema.PlayerCreate
    # or handle missing fields appropriately (e.g. with defaults or by raising an error).
    # For example, if 'team_id' is optional in PlayerCreate but required in models.Player (and not nullable),
    # this could cause issues.
    
    # Assuming player_schema.PlayerCreate includes all necessary fields or they are nullable/have defaults in models.Player
    db_player = models.Player(**player.model_dump())
    db.add(db_player)
    await db.commit()
    await db.refresh(db_player)
    return db_player 