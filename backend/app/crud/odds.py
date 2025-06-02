import uuid
from typing import List, Optional

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, and_
from uuid import UUID

from backend.db import models as db_models # Renamed to avoid conflict
from backend.schemas import odds as odds_schema # Use alias for Pydantic schemas

import sqlalchemy # Add this line to suppress Pylance import error if select is not found below
from sqlalchemy.ext.asyncio import AsyncSession # Add this
from sqlalchemy import select # Add this

async def get_game_odd(db: AsyncSession, game_odd_id: uuid.UUID) -> Optional[db_models.GameOdd]:
    stmt = (
        select(db_models.GameOdd)
        .options(
            joinedload(db_models.GameOdd.game),
            joinedload(db_models.GameOdd.bookmaker),
            joinedload(db_models.GameOdd.market),
        )
        .filter(db_models.GameOdd.id == game_odd_id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_game_odds(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    game_id: Optional[uuid.UUID] = None,
    # bookmaker_key: Optional[str] = None, # If filtering by bookmaker key is needed
    # market_key: Optional[str] = None, # If filtering by market key is needed
) -> List[db_models.GameOdd]:
    query = select(db_models.GameOdd).options(
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

    stmt = query.offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_player_prop(db: AsyncSession, player_prop_id: uuid.UUID) -> Optional[db_models.PlayerProp]:
    stmt = (
        select(db_models.PlayerProp)
        .options(
            joinedload(db_models.PlayerProp.game),
            joinedload(db_models.PlayerProp.player),
            joinedload(db_models.PlayerProp.bookmaker),
            joinedload(db_models.PlayerProp.market),
        )
        .filter(db_models.PlayerProp.id == player_prop_id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_player_props(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    game_id: Optional[uuid.UUID] = None,
    player_id: Optional[uuid.UUID] = None,
    # bookmaker_key: Optional[str] = None, # As above, for filtering by keys
    # market_key: Optional[str] = None,
) -> List[db_models.PlayerProp]:
    query = select(db_models.PlayerProp).options(
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

    stmt = query.offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

# We are not implementing create, update, delete here as the odds_scraper.py handles ingestion.
# If direct API manipulation of odds is needed later, those functions can be added. 