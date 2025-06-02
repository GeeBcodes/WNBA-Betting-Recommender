from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
import uuid # For UUID type
from typing import List, Optional
import logging # For logger in potential generate endpoint
from fastapi import status # Add this

from .. import dependencies as deps
from ..crud import parlays as crud_parlays
from schemas import parlay as parlay_schema # Alias for clarity
from backend.app.dependencies import get_db

# Get an instance of a logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/parlays",
    tags=["parlays"], # Changed to lowercase
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[parlay_schema.Parlay])
async def read_parlays_endpoint(
    skip: int = 0, 
    limit: int = 100, 
    min_combined_probability: Optional[float] = Query(None, ge=0, le=1),
    min_legs: Optional[int] = Query(None, ge=1),
    max_legs: Optional[int] = Query(None, ge=1),
    db: AsyncSession = Depends(get_db)
):
    parlays = await crud_parlays.get_parlays(
        db, 
        skip=skip, 
        limit=limit,
        min_combined_probability=min_combined_probability,
        min_legs=min_legs,
        max_legs=max_legs
    )
    return parlays

@router.get("/{parlay_id}", response_model=parlay_schema.Parlay)
async def read_parlay_endpoint(parlay_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    db_parlay = await crud_parlays.get_parlay(db, parlay_id=parlay_id)
    if db_parlay is None:
        raise HTTPException(status_code=404, detail="Parlay not found")
    return db_parlay

# Endpoint to trigger parlay generation as a background task
@router.post("/generate", status_code=202) # 202 Accepted
async def trigger_parlay_generation_endpoint(
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """
    Triggers the parlay generation process in the background.
    """
    # Import locally to avoid potential circular dependencies or heavy initial load
    from parlays import parlay_builder 
    
    logger.info("Received request to generate parlays.")
    # It's better to pass necessary, simple arguments to the task function
    # rather than the whole db session if the task is complex or might run much later.
    # However, for this pattern, SessionLocal() could be used within the task too.
    # For now, let's pass the session from the request, assuming it's okay for this use case.
    # Alternatively, parlay_builder.generate_and_store_parlays could create its own session.
    
    def build_parlays_task():
       # Create a new session for the background task to ensure thread safety / session scope
       task_db: AsyncSession = deps.SessionLocal()
       try:
           logger.info("Background task: Starting parlay generation.")
           parlay_builder.generate_and_store_parlays(task_db) # game_date can be passed or defaulted in generate_and_store_parlays
           logger.info("Background task: Parlay generation finished.")
       except Exception as e:
           logger.error(f"Background task: Error during parlay generation: {e}", exc_info=True)
       finally:
           task_db.close()

    background_tasks.add_task(build_parlays_task)
    return {"message": "Parlay generation process started in the background."}

@router.post("/", response_model=parlay_schema.Parlay, status_code=status.HTTP_201_CREATED)
async def create_parlay_endpoint(
    parlay: parlay_schema.ParlayCreate, 
    db: AsyncSession = Depends(get_db)
):
    # Basic validation example: ensure selections are not empty
    if not parlay.selections:
        raise HTTPException(status_code=400, detail="Parlay must include at least one selection.")
    
    # Further validation could be added here, e.g., checking if player_prop_ids in selections are valid
    # and if combined_probability and total_odds are consistent with selections (though these are often calculated on the frontend or by a service)

    return await crud_parlays.create_parlay(db=db, parlay=parlay)

# Add update/delete endpoints if/when CRUD functions are implemented
# Update might involve recalculating odds/probability if selections change. 

# Note: Creating parlays is currently handled by the parlay_builder.py script.
# If an endpoint for creating parlays via API is needed, it could be added.
# Example:
# @router.post("/", response_model=parlay_schema.Parlay, status_code=201)
# def create_new_parlay(
#     *, 
#     db: Session = Depends(deps.get_db), 
#     parlay_in: parlay_schema.ParlayCreate
# ):
#     return crud_parlays.create_parlay(db=db, parlay=parlay_in)

# Consider an endpoint to trigger parlay generation if it's not a scheduled task:
# @router.post("/generate", status_code=202) # 202 Accepted
# async def trigger_parlay_generation_endpoint(background_tasks: BackgroundTasks, db: Session = Depends(deps.get_db)):
#     from backend.parlays import parlay_builder # Avoid top-level import if heavy
#     def build_parlays_task():
#        logger.info("Background task: Starting parlay generation.")
#        parlay_builder.generate_and_store_parlays(db) # game_date can be passed or defaulted
#        logger.info("Background task: Parlay generation finished.")
#     background_tasks.add_task(build_parlays_task)
#     return {"message": "Parlay generation process started in the background."} 

# --- Placeholder for Parlay Management Endpoints (e.g., add_selection, remove_selection, update_status) ---
# These would require more complex logic and potentially new CRUD operations or service layer functions.

# Example: Update an existing parlay (e.g., its status or recalculate odds if a leg is voided)
# @router.put("/{parlay_id}", response_model=parlay_schema.Parlay)
# async def update_parlay_endpoint(
#     parlay_id: uuid.UUID, 
#     parlay_update: parlay_schema.ParlayUpdate, # Assuming a ParlayUpdate schema exists
#     db: AsyncSession = Depends(get_db)
# ):
#     # db_parlay = await crud.update_parlay(db, parlay_id=parlay_id, parlay_update=parlay_update)
#     # if db_parlay is None:
#     #     raise HTTPException(status_code=404, detail="Parlay not found or update failed")
#     # return db_parlay
#     raise HTTPException(status_code=501, detail="Update Parlay not implemented")


# Example: Delete a parlay
# @router.delete("/{parlay_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_parlay_endpoint(parlay_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
#     # success = await crud.delete_parlay(db, parlay_id=parlay_id)
#     # if not success:
#     #     raise HTTPException(status_code=404, detail="Parlay not found")
#     # return Response(status_code=status.HTTP_204_NO_CONTENT)
#     raise HTTPException(status_code=501, detail="Delete Parlay not implemented")

# --- Utility Endpoints for Parlay Calculation (Examples) ---

# @router.post("/calculate_parlay_odds/", response_model=parlay_schema.ParlayCalculationResponse)
# async def calculate_parlay_odds_endpoint(selections: List[parlay_schema.ParlaySelectionDetail]):
#     # This endpoint would likely call a service function to calculate combined odds and probability
#     # based on the provided selections. It might not directly interact with the DB for this calculation.
#     # Example: 
#     # calculated_data = ParlayService.calculate_combined_metrics(selections)
#     # return calculated_data
#     if not selections:
#         raise HTTPException(status_code=400, detail="Cannot calculate odds for empty selections.")
    
#     # Dummy calculation for illustration
#     calculated_total_odds = 1.0
#     for sel in selections:
#         # Assuming sel.odds is in American odds, convert to decimal, multiply, then convert back if needed
#         # This is a placeholder for actual odds calculation logic
#         if sel.get("odds") and isinstance(sel.get("odds"), (int, float)):
#             # Simplified: if odds are already decimal like 2.5, 1.8 etc. Otherwise conversion needed.
#             calculated_total_odds *= float(sel.get("odds")) 
#         else:
#             # Handle missing or invalid odds per selection
#             pass # Or raise error
            
#     # Placeholder for probability calculation
#     calculated_probability = 1 / calculated_total_odds if calculated_total_odds > 0 else 0

#     return parlay_schema.ParlayCalculationResponse(
#         selections=selections,
#         calculated_combined_probability=calculated_probability,
#         calculated_total_odds=calculated_total_odds
#     ) 