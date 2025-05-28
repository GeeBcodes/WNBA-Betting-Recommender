from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
import uuid # For UUID type
from typing import List, Optional
import logging # For logger in potential generate endpoint

from .. import dependencies as deps
from ..crud import parlays as crud_parlays
from schemas import parlay as parlay_schema # Alias for clarity

# Get an instance of a logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/parlays",
    tags=["parlays"], # Changed to lowercase
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=List[parlay_schema.Parlay])
def read_parlays(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = Query(default=100, ge=1, le=200),
    min_combined_probability: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum combined probability for the parlay."),
    min_legs: Optional[int] = Query(default=None, ge=2, description="Minimum number of legs in the parlay."),
    max_legs: Optional[int] = Query(default=None, ge=1, description="Maximum number of legs in the parlay.") # Corrected min_legs for max_legs here in description logic
):
    """
    Retrieve parlays.
    Supports pagination and filtering by minimum combined probability and number of legs.
    """
    if min_legs is not None and max_legs is not None and min_legs > max_legs:
        raise HTTPException(status_code=400, detail="min_legs cannot be greater than max_legs.")
        
    parlays = crud_parlays.get_parlays(
        db, 
        skip=skip, 
        limit=limit, 
        min_combined_probability=min_combined_probability,
        min_legs=min_legs,
        max_legs=max_legs
    )
    return parlays

@router.get("/{parlay_id}", response_model=parlay_schema.Parlay)
def read_parlay(
    parlay_id: uuid.UUID,
    db: Session = Depends(deps.get_db)
):
    """
    Retrieve a specific parlay by its UUID.
    """
    db_parlay = crud_parlays.get_parlay(db, parlay_id=parlay_id)
    if db_parlay is None:
        raise HTTPException(status_code=404, detail="Parlay not found")
    return db_parlay

# Endpoint to trigger parlay generation as a background task
@router.post("/generate", status_code=202) # 202 Accepted
async def trigger_parlay_generation_endpoint(
    background_tasks: BackgroundTasks, 
    db: Session = Depends(deps.get_db)
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
       task_db: Session = deps.SessionLocal()
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