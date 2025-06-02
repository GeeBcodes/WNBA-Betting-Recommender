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
    db_prediction = models.Prediction(
        player_prop_id=prediction.player_prop_id,
        model_version_id=prediction.model_version_id,
        predicted_value=prediction.predicted_value,
        predicted_over_probability=prediction.predicted_over_probability,
        predicted_under_probability=prediction.predicted_under_probability
        # prediction_datetime is default in model
    )
    db.add(db_prediction)
    await db.commit()
    await db.refresh(db_prediction)
    return await get_prediction(db, db_prediction.id)

async def get_prediction(db: AsyncSession, prediction_id: uuid.UUID) -> Optional[models.Prediction]:
    stmt = (
        select(models.Prediction)
        .options(
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.player),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.market),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.bookmaker),
            selectinload(models.Prediction.model_version)
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
        selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.player),
        selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game).selectinload(models.Game.home_team_ref),
        selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game).selectinload(models.Game.away_team_ref),
        selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.market),
        selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.bookmaker),
        selectinload(models.Prediction.model_version)
    )

    if game_date:
        query = query.join(models.PlayerProp, models.Prediction.player_prop_id == models.PlayerProp.id)\
                     .join(models.Game, models.PlayerProp.game_id == models.Game.id)\
                     .filter(models.Game.game_datetime >= game_date)
    
    if player_id:
        query = query.join(models.PlayerProp, models.Prediction.player_prop_id == models.PlayerProp.id)\
                     .filter(models.PlayerProp.player_id == player_id)

    if model_version_id:
        query = query.filter(models.Prediction.model_version_id == model_version_id)
        
    if bookmaker_key:
        query = query.join(models.PlayerProp, models.Prediction.player_prop_id == models.PlayerProp.id)\
                     .join(models.Bookmaker, models.PlayerProp.bookmaker_id == models.Bookmaker.id)\
                     .filter(models.Bookmaker.key == bookmaker_key)

    if market_key:
        query = query.join(models.PlayerProp, models.Prediction.player_prop_id == models.PlayerProp.id)\
                     .join(models.Market, models.PlayerProp.market_id == models.Market.id)\
                     .filter(models.Market.key == market_key)

    stmt = query.order_by(models.Prediction.prediction_datetime.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_predictions_by_player_prop(db: AsyncSession, player_prop_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[models.Prediction]:
    stmt = (
        select(models.Prediction)
        .options(
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.player),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.market),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.bookmaker),
            selectinload(models.Prediction.model_version)
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
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.player),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.game),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.market),
            selectinload(models.Prediction.player_prop).selectinload(models.PlayerProp.bookmaker),
            selectinload(models.Prediction.model_version)
        )
        .filter(models.Prediction.model_version_id == model_version_id)
        .order_by(models.Prediction.prediction_datetime.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all() 