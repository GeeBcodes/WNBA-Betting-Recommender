from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid # For UUID type
from typing import List

from backend.app import crud
from backend.schemas import parlay as parlay_schema # Alias for clarity
from backend.app.dependencies import get_db

router = APIRouter(
    prefix="/parlays",
    tags=["Parlays"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=parlay_schema.Parlay)
def create_parlay_endpoint(
    parlay: parlay_schema.ParlayCreate,
    db: Session = Depends(get_db)
):
    return crud.create_parlay(db=db, parlay=parlay)

@router.get("/", response_model=List[parlay_schema.Parlay])
def read_parlays_endpoint(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    # Optional: Add user_id filter if users are implemented
    parlays = crud.get_parlays(db, skip=skip, limit=limit)
    return parlays

@router.get("/{parlay_id}", response_model=parlay_schema.Parlay)
def read_parlay_endpoint(
    parlay_id: uuid.UUID, 
    db: Session = Depends(get_db)
):
    db_parlay = crud.get_parlay(db, parlay_id=parlay_id)
    if db_parlay is None:
        raise HTTPException(status_code=404, detail="Parlay not found")
    return db_parlay

# Add update/delete endpoints if/when CRUD functions are implemented
# Update might involve recalculating odds/probability if selections change. 