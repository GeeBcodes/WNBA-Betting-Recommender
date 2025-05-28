from sqlalchemy.orm import Session, joinedload # Added joinedload
import uuid # For UUID type
from typing import List, Optional # For list and optional types
from datetime import date # New import for date filtering

from backend.db import models
from backend.schemas import player as player_schema # New import
from backend.schemas import game as game_schema # New import
from backend.schemas import player_stats as player_stat_schema # New import for PlayerStat

# CRUD for ModelVersion - MOVED to crud/model_versions.py

# --- CRUD for Prediction --- MOVED to crud/predictions.py

# --- CRUD for Parlay --- MOVED to crud/parlays.py

# --- CRUD for Player ---
def get_player(db: Session, player_id: uuid.UUID) -> Optional[models.Player]:
    return db.query(models.Player).filter(models.Player.id == player_id).first()

def get_player_by_api_id(db: Session, player_api_id: str) -> Optional[models.Player]: # This might be wnba_player_id
    # Assuming player_api_id corresponds to wnba_player_id or a similar unique external ID
    # If models.Player has a specific field like `external_api_id` or `wnba_player_id`, use that.
    # For now, using a hypothetical `player_api_id` field.
    # Replace `models.Player.player_api_id` with the actual field name if different.
    # return db.query(models.Player).filter(models.Player.player_api_id == player_api_id).first()
    # Based on current models.Player, there is no player_api_id. Only player_name is unique.
    # This function might need revision based on how players are uniquely identified from APIs.
    # For now, let's assume player_name is used if no other API ID field exists.
    return db.query(models.Player).filter(models.Player.player_name == player_api_id).first() # Fallback to name if no API ID

def get_players(db: Session, skip: int = 0, limit: int = 100) -> List[models.Player]:
    return db.query(models.Player).offset(skip).limit(limit).all()

def create_player(db: Session, player: player_schema.PlayerCreate) -> models.Player:
    db_player = models.Player(**player.model_dump())
    db.add(db_player)
    db.commit()
    db.refresh(db_player)
    return db_player

# --- CRUD for Game ---
def get_game(db: Session, game_id: uuid.UUID) -> Optional[models.Game]:
    return db.query(models.Game).filter(models.Game.id == game_id).first()

def get_game_by_external_id(db: Session, external_id: str) -> Optional[models.Game]:
    # Assuming Game model has an 'external_id' field for IDs from data sources like sportsdataverse
    return db.query(models.Game).filter(models.Game.external_id == external_id).first()

def get_games(db: Session, skip: int = 0, limit: int = 100) -> List[models.Game]:
    return db.query(models.Game).order_by(models.Game.game_datetime.desc()).offset(skip).limit(limit).all()

def create_game(db: Session, game: game_schema.GameCreate) -> models.Game:
    db_game = models.Game(**game.model_dump())
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game

# --- CRUD for PlayerStat --- MOVED to crud/player_stats.py
# The get_player_stat, get_player_stats, create_player_stat functions were here.
# We are now relying on the versions in crud/player_stats.py 