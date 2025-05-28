import uuid
from typing import List, Optional
from datetime import date

from sqlalchemy.orm import Session, joinedload

from db import models as db_models
# Schemas are not directly used here but might be for create/update later
# from backend.schemas import player_stats as player_stat_schemas 

def get_player_stat(db: Session, player_stat_id: uuid.UUID) -> Optional[db_models.PlayerStat]:
    """Fetches a single player statistic record by its ID, with related player and game."""
    return (
        db.query(db_models.PlayerStat)
        .options(
            joinedload(db_models.PlayerStat.player),
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.home_team_ref),
            joinedload(db_models.PlayerStat.game).joinedload(db_models.Game.away_team_ref),
        )
        .filter(db_models.PlayerStat.id == player_stat_id)
        .first()
    )

def get_player_stats(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    player_id: Optional[uuid.UUID] = None,
    game_id: Optional[uuid.UUID] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[db_models.PlayerStat]:
    """Fetches multiple player statistic records with filtering and pagination."""
    query = db.query(db_models.PlayerStat).options(
        joinedload(db_models.PlayerStat.player), # Eager load player data
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

    return query.offset(skip).limit(limit).all()

# Create, Update, Delete functions for PlayerStat can be added here if needed for API endpoints
# For now, data is ingested by wnba_stats_scraper.py 