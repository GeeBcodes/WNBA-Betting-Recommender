from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
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

@router.post("/", response_model=prediction_schema.Prediction)
def create_prediction_endpoint(
    prediction: prediction_schema.PredictionCreate,
    db: Session = Depends(deps.get_db)
):
    return crud_predictions.create_prediction(db=db, prediction=prediction)

@router.get("/", response_model=List[prediction_schema.Prediction])
def read_predictions(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = Query(default=100, ge=1, le=500), # Added validation for limit
    game_date: Optional[date] = Query(default=None, description="Filter by game date (YYYY-MM-DD). Predictions for games on or after this date."),
    player_id: Optional[uuid.UUID] = Query(default=None, description="Filter by player UUID."),
    model_version_id: Optional[uuid.UUID] = Query(default=None, description="Filter by model version UUID."),
    bookmaker_key: Optional[str] = Query(default=None, description="Filter by bookmaker key (e.g., 'bovada', 'prizepicks')."),
    market_key: Optional[str] = Query(default=None, description="Filter by market key (e.g., 'player_points', 'player_rebounds').")
):
    """
    Retrieve predictions.
    Supports pagination and filtering by game_date, player_id, model_version_id, bookmaker_key, and market_key.
    """
    predictions = crud_predictions.get_predictions(
        db,
        skip=skip, 
        limit=limit, 
        game_date=game_date,
        player_id=player_id,
        model_version_id=model_version_id,
        bookmaker_key=bookmaker_key,
        market_key=market_key
    )
    return predictions

@router.get("/{prediction_id}", response_model=prediction_schema.Prediction)
def read_prediction(
    prediction_id: uuid.UUID,
    db: Session = Depends(deps.get_db)
):
    """
    Retrieve a specific prediction by its UUID.
    """
    db_prediction = crud_predictions.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction

@router.get("/by_prop/{player_prop_id}", response_model=List[prediction_schema.Prediction])
def read_predictions_by_prop_endpoint(
    player_prop_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db)
):
    predictions = crud_predictions.get_predictions_by_player_prop(db, player_prop_id=player_prop_id, skip=skip, limit=limit)
    return predictions

@router.get("/by_model_version/{model_version_id}", response_model=List[prediction_schema.Prediction])
def read_predictions_by_model_version_endpoint(
    model_version_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db)
):
    predictions = crud_predictions.get_predictions_by_model_version(db, model_version_id=model_version_id, skip=skip, limit=limit)
    return predictions

# Add update/delete endpoints if/when CRUD functions are implemented 