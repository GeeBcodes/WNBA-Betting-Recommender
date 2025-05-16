from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid # For UUID type
from typing import List

from backend.app import crud
from backend.schemas import prediction as prediction_schema # Alias for clarity
from backend.app.dependencies import get_db

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=prediction_schema.Prediction)
def create_prediction_endpoint(
    prediction: prediction_schema.PredictionCreate,
    db: Session = Depends(get_db)
):
    
    return crud.create_prediction(db=db, prediction=prediction)

@router.get("/", response_model=List[prediction_schema.Prediction])
def read_predictions_endpoint(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = crud.get_predictions(db, skip=skip, limit=limit)
    return predictions

@router.get("/{prediction_id}", response_model=prediction_schema.Prediction)
def read_prediction_endpoint(
    prediction_id: uuid.UUID, 
    db: Session = Depends(get_db)
):
    db_prediction = crud.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction

@router.get("/by_prop_odd/{player_prop_odd_id}", response_model=List[prediction_schema.Prediction])
def read_predictions_by_prop_odd_endpoint(
    player_prop_odd_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = crud.get_predictions_by_player_prop_odd(db, player_prop_odd_id=player_prop_odd_id, skip=skip, limit=limit)
    
    return predictions

@router.get("/by_model_version/{model_version_id}", response_model=List[prediction_schema.Prediction])
def read_predictions_by_model_version_endpoint(
    model_version_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    predictions = crud.get_predictions_by_model_version(db, model_version_id=model_version_id, skip=skip, limit=limit)
   
    return predictions

# Add update/delete endpoints if/when CRUD functions are implemented 