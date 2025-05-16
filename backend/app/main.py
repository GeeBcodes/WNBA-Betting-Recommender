from fastapi import FastAPI
from backend.schemas.health import PingResponse
from backend.app.routers import player_stats, odds # Corrected import path for consistency
from backend.app.routers import model_versions, predictions, parlays # Corrected import path for consistency

app = FastAPI(
    title="WNBA Betting Recommender API",
    version="0.1.0",
)

app.include_router(player_stats.router)
app.include_router(odds.router)
app.include_router(model_versions.router) # Include ModelVersions router
app.include_router(predictions.router)    # Include Predictions router
app.include_router(parlays.router)        # Include Parlays router

@app.get("/", tags=["Root"])
async def get_root():
    """Welcome message for the API root."""
    return {"message": "Welcome to the WNBA Betting Recommender API!"}

@app.get("/api/ping", response_model=PingResponse, tags=["Health"])
async def ping_api():
    """Simple ping to check API health."""
    return {"status": "ok", "message": "pong"} 