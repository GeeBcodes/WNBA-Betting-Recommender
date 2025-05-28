import uuid
from typing import List, Optional

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, and_
from uuid import UUID

from db import models as db_models # Renamed to avoid conflict
from schemas import odds as odds_schema # Use alias for Pydantic schemas

def get_game_odd(db: Session, game_odd_id: uuid.UUID) -> Optional[db_models.GameOdd]:
    return (
        db.query(db_models.GameOdd)
        .options(
            joinedload(db_models.GameOdd.game),
            joinedload(db_models.GameOdd.bookmaker),
            joinedload(db_models.GameOdd.market),
        )
        .filter(db_models.GameOdd.id == game_odd_id)
        .first()
    )

def get_game_odds(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    game_id: Optional[uuid.UUID] = None,
    # bookmaker_key: Optional[str] = None, # If filtering by bookmaker key is needed
    # market_key: Optional[str] = None, # If filtering by market key is needed
) -> List[db_models.GameOdd]:
    query = db.query(db_models.GameOdd).options(
        joinedload(db_models.GameOdd.game),
        joinedload(db_models.GameOdd.bookmaker),
        joinedload(db_models.GameOdd.market),
    )

    if game_id:
        query = query.filter(db_models.GameOdd.game_id == game_id)
    
    # Add filters for bookmaker_key and market_key if relationships are set up for direct query
    # For example, if Bookmaker model is imported as db_models.Bookmaker:
    # if bookmaker_key:
    # query = query.join(db_models.Bookmaker).filter(db_models.Bookmaker.key == bookmaker_key)
    # if market_key:
    # query = query.join(db_models.Market).filter(db_models.Market.key == market_key)

    return query.offset(skip).limit(limit).all()

def get_player_prop(db: Session, player_prop_id: uuid.UUID) -> Optional[db_models.PlayerProp]:
    return (
        db.query(db_models.PlayerProp)
        .options(
            joinedload(db_models.PlayerProp.game),
            joinedload(db_models.PlayerProp.player),
            joinedload(db_models.PlayerProp.bookmaker),
            joinedload(db_models.PlayerProp.market),
        )
        .filter(db_models.PlayerProp.id == player_prop_id)
        .first()
    )

def get_player_props(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    game_id: Optional[uuid.UUID] = None,
    player_id: Optional[uuid.UUID] = None,
    # bookmaker_key: Optional[str] = None, # As above, for filtering by keys
    # market_key: Optional[str] = None,
) -> List[db_models.PlayerProp]:
    query = db.query(db_models.PlayerProp).options(
        joinedload(db_models.PlayerProp.game),
        joinedload(db_models.PlayerProp.player),
        joinedload(db_models.PlayerProp.bookmaker),
        joinedload(db_models.PlayerProp.market),
    )

    if game_id:
        query = query.filter(db_models.PlayerProp.game_id == game_id)
    if player_id:
        query = query.filter(db_models.PlayerProp.player_id == player_id)

    # Add filters for bookmaker_key and market_key as in get_game_odds

    return query.offset(skip).limit(limit).all()

# We are not implementing create, update, delete here as the odds_scraper.py handles ingestion.
# If direct API manipulation of odds is needed later, those functions can be added. 