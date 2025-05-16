from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid # For UUID type
from typing import List

from backend.app import crud
from backend.schemas import model_version as model_version_schema # Alias for clarity
from backend.app.dependencies import get_db

router = APIRouter(
    prefix="/model_versions",
    tags=["Model Versions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=model_version_schema.ModelVersion)
def create_model_version_endpoint(
    model_version: model_version_schema.ModelVersionCreate,
    db: Session = Depends(get_db)
):
    # Optional: Check if model version with the same name already exists
    # db_model_version_by_name = crud.get_model_version_by_name(db, version_name=model_version.version_name)
    # if db_model_version_by_name:
    #     raise HTTPException(status_code=400, detail="Model version name already registered")
    return crud.create_model_version(db=db, model_version=model_version)

@router.get("/", response_model=List[model_version_schema.ModelVersion])
def read_model_versions_endpoint(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    model_versions = crud.get_model_versions(db, skip=skip, limit=limit)
    return model_versions

@router.get("/{model_version_id}", response_model=model_version_schema.ModelVersion)
def read_model_version_endpoint(
    model_version_id: uuid.UUID, 
    db: Session = Depends(get_db)
):
    db_model_version = crud.get_model_version(db, model_version_id=model_version_id)
    if db_model_version is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return db_model_version

@router.get("/name/{version_name}", response_model=model_version_schema.ModelVersion)
def read_model_version_by_name_endpoint(
    version_name: str, 
    db: Session = Depends(get_db)
):
    db_model_version = crud.get_model_version_by_name(db, version_name=version_name)
    if db_model_version is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    return db_model_version

# Add update/delete endpoints if/when CRUD functions are implemented and requirements are clear 