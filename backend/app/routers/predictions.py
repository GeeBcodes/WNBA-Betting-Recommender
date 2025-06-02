from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import uuid # For UUID type
from typing import List, Optional
from datetime import date

from .. import dependencies as deps
from ..crud import predictions as crud_predictions
from schemas import prediction as prediction_schema # Use prediction_schema to avoid name clashes

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=prediction_schema.Prediction, status_code=status.HTTP_201_CREATED)
async def create_prediction_endpoint(
    prediction: prediction_schema.PredictionCreate,
    db: AsyncSession = Depends(deps.get_db)
):
    # Optional: Add validation here, e.g., check if player_prop_id and model_version_id exist
    # db_player_prop = await crud.get_player_prop(db, prediction.player_prop_id) # Assuming a general get_player_prop exists
    # if not db_player_prop:
    #     raise HTTPException(status_code=404, detail=f"PlayerProp with id {prediction.player_prop_id} not found")
    
    # db_model_version = await crud.get_model_version(db, prediction.model_version_id)
    # if not db_model_version:
    #     raise HTTPException(status_code=404, detail=f"ModelVersion with id {prediction.model_version_id} not found")

    return await crud_predictions.create_prediction(db=db, prediction=prediction)

@router.get("/", response_model=List[prediction_schema.Prediction])
async def read_predictions_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    game_date: Optional[date] = None,
    player_id: Optional[uuid.UUID] = None,
    model_version_id: Optional[uuid.UUID] = None,
    bookmaker_key: Optional[str] = None,
    market_key: Optional[str] = None,
    db: AsyncSession = Depends(deps.get_db)
):
    predictions = await crud_predictions.get_predictions(
        db, skip=skip, limit=limit, game_date=game_date, 
        player_id=player_id, model_version_id=model_version_id,
        bookmaker_key=bookmaker_key, market_key=market_key
    )
    return predictions

@router.get("/{prediction_id}", response_model=prediction_schema.Prediction)
async def read_prediction_endpoint(prediction_id: uuid.UUID, db: AsyncSession = Depends(deps.get_db)):
    db_prediction = await crud_predictions.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction

@router.get("/by_player_prop/{player_prop_id}", response_model=List[prediction_schema.Prediction])
async def read_predictions_by_player_prop_endpoint(
    player_prop_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(deps.get_db)
):
    predictions = await crud_predictions.get_predictions_by_player_prop(db, player_prop_id=player_prop_id, skip=skip, limit=limit)
    # if not predictions:
    #     raise HTTPException(status_code=404, detail="No predictions found for this player prop") # Optional: or return empty list
    return predictions

@router.get("/by_model_version/{model_version_id}", response_model=List[prediction_schema.Prediction])
async def read_predictions_by_model_version_endpoint(
    model_version_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(deps.get_db)
):
    predictions = await crud_predictions.get_predictions_by_model_version(db, model_version_id=model_version_id, skip=skip, limit=limit)
    # if not predictions:
    #     raise HTTPException(status_code=404, detail="No predictions found for this model version") # Optional: or return empty list
    return predictions

# Add update/delete endpoints if/when CRUD functions are implemented 