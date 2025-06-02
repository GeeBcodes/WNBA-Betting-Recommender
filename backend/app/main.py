import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.health import PingResponse
from .routers import player_stats, odds
from .routers import model_versions, predictions, parlays

app = FastAPI(
    title="WNBA Betting Recommender API",
    description="API for WNBA game data, odds, predictions, and parlay recommendations.",
    version="0.1.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # Frontend URL for Vite
    "http://localhost:3000",  # In case create-react-app default is used later
    # Add any other origins you need to allow (e.g., production frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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