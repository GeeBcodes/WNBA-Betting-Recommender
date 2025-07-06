import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session

from backend.db import models
from backend.schemas import game as game_schema

# --- CRUD for Game ---
async def get_game(db: AsyncSession, game_id: uuid.UUID) -> Optional[models.Game]:
    stmt = select(models.Game).filter(models.Game.id == game_id)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_game_by_external_id(db: AsyncSession, external_id: int) -> Optional[models.Game]:
    # Assuming Game model has an 'external_id' field for IDs from data sources like sportsdataverse
    stmt = select(models.Game).filter(models.Game.the_odds_api_game_id == str(external_id))
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_games(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[models.Game]:
    stmt = select(models.Game).order_by(models.Game.game_datetime.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_game(db: AsyncSession, game: game_schema.GameCreate) -> models.Game:
    # Create a dictionary with only the fields that are valid for the Game model
    game_data = {
        'the_odds_api_game_id': game.the_odds_api_game_id,
        'game_datetime': game.game_datetime,
        'season': game.season,
        'status': game.status,
        'home_team_id': game.home_team_id,
        'away_team_id': game.away_team_id,
        'home_score': game.home_team_score,  # Schema uses home_team_score
        'away_score': game.away_team_score,  # Schema uses away_team_score
        'home_team_possessions': game.home_team_possessions,
        'away_team_possessions': game.away_team_possessions,
    }
    
    # Filter out None values so we can rely on database defaults
    db_game_data = {k: v for k, v in game_data.items() if v is not None}
    
    db_game = models.Game(**db_game_data)
    db.add(db_game)
    await db.commit()
    await db.refresh(db_game)
    return db_game

# --- Synchronous CRUD for use in scripts ---
def create_game_sync(db: Session, game: game_schema.GameCreate) -> models.Game:
    """Synchronous version of create_game."""
    game_data = {
        'the_odds_api_game_id': game.the_odds_api_game_id,
        'game_datetime': game.game_datetime,
        'season': game.season,
        'status': game.status,
        'home_team_id': game.home_team_id,
        'away_team_id': game.away_team_id
    }
    db_game_data = {k: v for k, v in game_data.items() if v is not None}
    
    db_game = models.Game(**db_game_data)
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game 