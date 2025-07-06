from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import uuid # For UUID type
from typing import List, Optional
from datetime import date
import logging

from backend.app import dependencies as deps
from backend.app.crud import predictions as crud_predictions
from backend.app.crud import player_props as crud_player_props # Import the new CRUD module
from backend.schemas import prediction as prediction_schema # Use prediction_schema to avoid name clashes
from backend.predict import predictor

# Setup logger for this router
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/run/{game_id}", response_model=List[prediction_schema.PredictionRead])
async def run_prediction_for_game_endpoint(
    game_id: uuid.UUID,
    db: AsyncSession = Depends(deps.get_db)
):
    """
    Run the predictor for a specific game and store the results.
    """
    try:
        # Step 1: Fetch the relevant player props for the game.
        player_props = await crud_player_props.get_player_props_by_game_id(db, game_id=game_id)
        
        if not player_props:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No player props found for the game, so no predictions were generated."
            )

        # Step 2: Call the correct predictor function with the props.
        # This function will save the predictions to the database.
        await predictor.make_predictions_for_props(db, player_props)
        
        # Step 3: Fetch the newly created predictions to return them.
        # We can identify them by the player_prop_ids we started with.
        player_prop_ids = [prop.id for prop in player_props]
        
        # This is a simplified fetch. A more robust way might be to filter by model_version
        # and a recent timestamp if `get_predictions_by_player_prop` isn't suitable.
        # For now, we assume we want all predictions for these props.
        
        # A simple way to get all predictions for a game is to just... get all predictions for that game.
        # Let's add a `get_predictions_by_game_id` to crud_predictions.
        # For now, let's just return a success message as the predictions are in the db.
        
        # Let's re-fetch the predictions for the props we processed
        newly_created_predictions = await crud_predictions.get_predictions_by_prop_ids(db, player_prop_ids)

        return newly_created_predictions

    except HTTPException as http_exc:
        # Re-raise HTTPException directly to let FastAPI handle it as the response
        raise http_exc
    except Exception as e:
        # For any other unexpected error, log it and return a 500
        logger.error(f"An unexpected error occurred during prediction for game {game_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A critical unexpected error occurred: {str(e)}"
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

@router.get("/", response_model=List[prediction_schema.PredictionRead])
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

@router.get("/{prediction_id}", response_model=prediction_schema.PredictionRead)
async def read_prediction_endpoint(prediction_id: uuid.UUID, db: AsyncSession = Depends(deps.get_db)):
    db_prediction = await crud_predictions.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction

@router.get("/by_player_prop/{player_prop_id}", response_model=List[prediction_schema.PredictionRead])
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

@router.get("/by_model_version/{model_version_id}", response_model=List[prediction_schema.PredictionRead])
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