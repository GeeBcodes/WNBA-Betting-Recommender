import uuid
from typing import List, Optional
from datetime import date
import logging # Added for logging
import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession # Add this
from sqlalchemy import select # Add this

from sqlalchemy.orm import Session, joinedload

from backend.db import models as db_models
# Schemas are not directly used here but might be for create/update later
# from backend.schemas import player_stats as player_stat_schemas 

logger = logging.getLogger(__name__) # Added for logging

async def get_player_stat(db: AsyncSession, player_stat_id: uuid.UUID) -> Optional[db_models.PlayerStat]:
    """Fetches a single player statistic record by its ID, with related player and game."""
    stmt = (
        select(db_models.PlayerStat)
        .options(
            joinedload(db_models.PlayerStat.player).joinedload(db_models.Player.team),
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.home_team_ref),
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.away_team_ref),
        )
        .filter(db_models.PlayerStat.id == player_stat_id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_player_stats(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    player_id: Optional[uuid.UUID] = None,
    game_id: Optional[uuid.UUID] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[db_models.PlayerStat]:
    """Fetches multiple player statistic records with filtering and pagination."""
    logger.info("Fetching player stats with eager loading...") # Added log
    query = select(db_models.PlayerStat).options(
        joinedload(db_models.PlayerStat.player).joinedload(db_models.Player.team), # Eager load player and their team
        joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.home_team_ref),   # Eager load game data and its home_team_ref
        joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.away_team_ref),   # Eager load game data and its away_team_ref
    )

    if player_id:
        query = query.filter(db_models.PlayerStat.player_id == player_id)
    if game_id:
        query = query.filter(db_models.PlayerStat.game_id == game_id)
    if start_date:
        query = query.filter(db_models.PlayerStat.game_date >= start_date)
    if end_date:
        query = query.filter(db_models.PlayerStat.game_date <= end_date)
    
    # Add sorting if desired, e.g., by game_date then player_name
    # query = query.order_by(db_models.PlayerStat.game_date.desc(), db_models.Player.player_name)
    # Requires joining with Player model if sorting by player_name and player relationship isn't already covering it for sort

    stmt = query.offset(skip).limit(limit)
    result = await db.execute(stmt)
    results = result.scalars().all()

    # Log details of the first result, if any, to check loaded relationships
    if results:
        first_res = results[0]
        logger.info(f"First player_stat ID: {first_res.id}")
        if first_res.player:
            logger.info(f"  Player ID: {first_res.player.id}, Name: {first_res.player.player_name}")
            if first_res.player.team:
                logger.info(f"    Player's Team ID: {first_res.player.team.id}, Team Name: {first_res.player.team.team_name}")
            else:
                logger.info(f"    Player's Team relationship is None (team_id: {first_res.player.team_id})")
        else:
            logger.info(f"  Player relationship is None (player_id: {first_res.player_id})")
        
        if first_res.game:
            logger.info(f"  Game ID: {first_res.game.id}, External ID: {first_res.game.external_id}")
            if first_res.game.home_team_ref:
                logger.info(f"    Game Home Team ID: {first_res.game.home_team_ref.id}, Name: {first_res.game.home_team_ref.team_name}")
            else:
                logger.info(f"    Game Home Team Ref is None (home_team_id: {first_res.game.home_team_id})")
            if first_res.game.away_team_ref:
                logger.info(f"    Game Away Team ID: {first_res.game.away_team_ref.id}, Name: {first_res.game.away_team_ref.team_name}")
            else:
                logger.info(f"    Game Away Team Ref is None (away_team_id: {first_res.game.away_team_id})")
        else:
            logger.info(f"  Game relationship is None (game_id: {first_res.game_id})")
    else:
        logger.info("No player stats found with current filters.")

    return results

# Create, Update, Delete functions for PlayerStat can be added here if needed for API endpoints
# For now, data is ingested by wnba_stats_scraper.py 