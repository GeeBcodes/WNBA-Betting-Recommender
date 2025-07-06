import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload
import uuid
from typing import List, Optional
from datetime import date

from backend.db import models
from backend.schemas import prediction as prediction_schema

# --- CRUD for Prediction ---
async def create_prediction(db: AsyncSession, prediction: prediction_schema.PredictionCreate) -> models.Prediction:
    # Use model_dump() to get a dictionary of all fields from the Pydantic schema
    prediction_data = prediction.model_dump()

    # Create the SQLAlchemy model instance from the dictionary
    db_prediction = models.Prediction(**prediction_data)
    
    db.add(db_prediction)
    await db.commit()
    await db.refresh(db_prediction)
    # The get_prediction call will ensure all relationships are eagerly loaded for the response
    return await get_prediction(db, db_prediction.id)

async def create_predictions_bulk(db: AsyncSession, predictions: List[prediction_schema.PredictionCreate]) -> List[models.Prediction]:
    """
    Creates multiple prediction records in the database in a single transaction.
    """
    db_predictions = [models.Prediction(**p.model_dump()) for p in predictions]
    db.add_all(db_predictions)
    await db.commit()
    
    # Refresh each object to get database-assigned values like IDs and defaults
    for db_prediction in db_predictions:
        await db.refresh(db_prediction)
        
    # Now that they are refreshed, we can load relationships
    # We'll fetch them again with relationships loaded to match the return type of other functions
    prediction_ids = [p.id for p in db_predictions]
    stmt = (
        select(models.Prediction)
        .options(
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.player).selectinload(models.Player.team),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game).selectinload(models.Game.home_team_ref),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game).selectinload(models.Game.away_team_ref),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.market),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.bookmaker),
            selectinload(models.Prediction.model_version)
        )
        .filter(models.Prediction.id.in_(prediction_ids))
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_prediction(db: AsyncSession, prediction_id: uuid.UUID) -> Optional[models.Prediction]:
    stmt = (
        select(models.Prediction)
        .options(
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.player).joinedload(models.Player.team),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.market),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.bookmaker),
            joinedload(models.Prediction.model_version)
        )
        .filter(models.Prediction.id == prediction_id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_predictions(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100, 
    game_date: Optional[date] = None,
    player_id: Optional[uuid.UUID] = None,
    model_version_id: Optional[uuid.UUID] = None,
    bookmaker_key: Optional[str] = None,
    market_key: Optional[str] = None
) -> List[models.Prediction]:
    query = select(models.Prediction).options(
        joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
        joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
        joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.player),
        joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.market),
        joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.bookmaker),
        joinedload(models.Prediction.model_version_reg),
        joinedload(models.Prediction.model_version_clf)
    )

    # Join related tables for filtering
    query = query.join(models.Prediction.player_prop)

    if game_date:
        query = query.join(models.PlayerProp.game).filter(sqlalchemy.func.date(models.Game.game_datetime) >= game_date)
    
    if player_id:
        query = query.filter(models.PlayerProp.player_id == player_id)

    if model_version_id:
        query = query.filter(models.Prediction.model_version_id == model_version_id)
        
    if bookmaker_key:
        query = query.join(models.PlayerProp.bookmaker).filter(models.Bookmaker.key == bookmaker_key)

    if market_key:
        query = query.join(models.PlayerProp.market).filter(models.Market.key == market_key)

    stmt = query.order_by(models.Prediction.prediction_datetime.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().unique().all()

async def get_predictions_by_player_prop(db: AsyncSession, player_prop_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    stmt = (
        select(models.Prediction)
        .options(
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.player).joinedload(models.Player.team),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.market),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.bookmaker),
            joinedload(models.Prediction.model_version)
        )
        .filter(models.Prediction.player_prop_id == player_prop_id)
        .order_by(models.Prediction.prediction_datetime.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_predictions_by_model_version(db: AsyncSession, model_version_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    stmt = (
        select(models.Prediction)
        .options(
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.player).joinedload(models.Player.team),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.market),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.bookmaker),
            joinedload(models.Prediction.model_version)
        )
        .filter(models.Prediction.model_version_id == model_version_id)
        .order_by(models.Prediction.prediction_datetime.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_predictions_by_prop_ids(db: AsyncSession, player_prop_ids: List[uuid.UUID]) -> List[models.Prediction]:
    """
    Fetches all predictions for a given list of player_prop_ids.
    """
    if not player_prop_ids:
        return []
    
    stmt = (
        select(models.Prediction)
        .options(
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.player).joinedload(models.Player.team),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.home_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.game).joinedload(models.Game.away_team_ref),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.market),
            joinedload(models.Prediction.player_prop).joinedload(models.PlayerProp.bookmaker),
            joinedload(models.Prediction.model_version)
        )
        .filter(models.Prediction.player_prop_id.in_(player_prop_ids))
        .order_by(models.Prediction.prediction_datetime.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def create_or_update_prediction(db: AsyncSession, prediction_data: prediction_schema.PredictionCreate) -> models.Prediction:
    """
    Creates a new prediction or updates an existing one based on player_prop_id and model_version_id.
    """
    stmt = select(models.Prediction).filter_by(
        player_prop_id=prediction_data.player_prop_id,
        model_version_id=prediction_data.model_version_id
    )
    result = await db.execute(stmt)
    db_prediction = result.scalars().first()

    prediction_data_dict = prediction_data.model_dump(exclude_unset=True)

    if db_prediction:
        # Update existing prediction
        for key, value in prediction_data_dict.items():
            setattr(db_prediction, key, value)
    else:
        # Create new prediction
        db_prediction = models.Prediction(**prediction_data_dict)
        db.add(db_prediction)

    await db.commit()
    await db.refresh(db_prediction)
    return db_prediction 