import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List, Optional
import logging

from backend.db import models
from backend.schemas import player as player_schema
from backend.app.crud.teams import get_team_by_api_id, create_team

logger = logging.getLogger(__name__)

# --- CRUD for Player ---
async def get_player(db: AsyncSession, player_id: uuid.UUID) -> Optional[models.Player]:
    stmt = select(models.Player).filter(models.Player.id == player_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_player_by_api_id(db: AsyncSession, player_api_id: str) -> Optional[models.Player]:
    """Fetches a player by their unique external API ID."""
    stmt = select(models.Player).filter(models.Player.api_player_id == player_api_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def create_player_from_dict(db: AsyncSession, player_data: dict) -> Optional[models.Player]:
    """
    Creates a new player from a dictionary of data, or updates an existing player's team.
    Intended for use by scrapers.
    """
    logger.debug(f"Processing player data with keys: {list(player_data.keys())}")
    
    api_player_id = str(player_data.get('athlete_id'))
    if not api_player_id or api_player_id == 'nan':
        logger.warning(f"Player creation skipped: 'athlete_id' is missing or invalid. Value: '{api_player_id}'")
        return None

    player_name = player_data.get('athlete_display_name')
    if not player_name:
        logger.warning(f"Player creation skipped for athlete_id '{api_player_id}': 'athlete_display_name' is missing.")
        return None

    # Determine team from the data
    api_team_id = str(player_data.get('team_id'))
    if not api_team_id:
        logger.warning(f"Player creation skipped for '{player_name}': 'team_id' is missing.")
        return None
        
    team_in_db = await get_team_by_api_id(db, api_team_id)
    if not team_in_db:
        # This part handles cases where the team from the player's box score doesn't exist yet
        team_data_for_creation = {
            'api_team_id': api_team_id,
            'name': player_data.get('team_abbreviation', f"Team {api_team_id}") 
        }
        team_in_db = await create_team(db, team_data_for_creation)
    
    team_id_for_player = team_in_db.id if team_in_db else None

    # Check for existing player
    existing_player = await get_player_by_api_id(db, api_player_id)
    
    if existing_player:
        # Player exists, check if team needs updating
        if team_id_for_player and existing_player.team_id != team_id_for_player:
            logger.info(f"Player '{player_name}' team change detected. Staging update for team ID from {existing_player.team_id} to {team_id_for_player}.")
            existing_player.team_id = team_id_for_player
            # NO COMMIT HERE. The calling function is responsible for the session's transaction.
            # db.add(existing_player) is not strictly necessary if the object is already in the session,
            # but it doesn't hurt.
            db.add(existing_player)
        return existing_player

    # Player does not exist, create new one
    logger.info(f"Creating new player: {player_name} with team ID {team_id_for_player}")
    new_player = models.Player(
        api_player_id=api_player_id, 
        player_name=player_name,
        team_id=team_id_for_player
    )
    
    db.add(new_player)
    # Flush the session to get the new player's ID without committing the transaction.
    await db.flush()
    await db.refresh(new_player)
    
    return new_player

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