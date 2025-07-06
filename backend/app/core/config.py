# backend/app/core/config.py

import os
from typing import List, Dict, Optional

from pydantic_settings import BaseSettings

# --- FIX: Robustly locate the .env file in the PROJECT ROOT directory ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")
# --- END FIX ---


class Settings(BaseSettings):
    """Application settings."""
    database_url: str
    echo_sql: bool = True
    test_database_url: Optional[str] = None
    sentry_dsn: Optional[str] = None

    # Odds API
    the_odds_api_key: Optional[str] = None

    # SportsDataVerse API
    sportsdataverse_api_key: Optional[str] = None


    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()

# --- Seasons ---
# Default seasons to use for training if none are provided
DEFAULT_SEASONS: List[int] = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# --- Model Training ---
# The list of player statistics we want to train models for.
TARGET_STATS_TO_TRAIN: List[str] = [
    "points",
    "rebounds",
    "assists",
    "three_pointers_made",
    "pra",  # Points + Rebounds + Assists
    "pr",   # Points + Rebounds
    "pa",   # Points + Assists
    "ra",   # Rebounds + Assists
    "blocks_plus_steals"
]

# For regression models, specifies whether to apply a Box-Cox transformation
# to the target variable to stabilize variance and handle non-normality.
TARGET_TRANSFORM_FOR_REGRESSION: Dict[str, bool] = {
    "points": True,
    "rebounds": True,
    "assists": True,
    "three_pointers_made": True,
    "pra": True,
    "pr": True,
    "pa": True,
    "ra": True,
    "blocks_plus_steals": True
}

# Rolling window sizes (in games) to use for generating player and team features.
ROLLING_WINDOWS: List[int] = [3, 5, 10, 20]


# --- Prop Market Mappings ---
# This dictionary maps the standardized statistic names used in our database
# to the often inconsistent market names used by betting odds providers.
# This is crucial for looking up the correct betting lines for a given stat.
DEFAULT_PROP_MARKET_TO_STAT_MAP: Dict[str, str] = {
    # Keys confirmed from the debug log
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr",
    "player_points_assists": "pa",
    "player_rebounds_assists": "ra",

    # Other potential keys based on the pattern
    "player_threes": "three_pointers_made",
    "player_blocks_steals": "blocks_plus_steals",

    # --- Keep these as fallbacks for other data sources ---
    "player_points_over_under": "points",
    "player_rebounds_over_under": "rebounds",
    "player_assists_over_under": "assists",
    "player_threes_over_under": "three_pointers_made",
    "player_blocks_steals_over_under": "blocks_plus_steals",
    "player_pts_rebs_asts_over_under": "pra",
    "player_points_rebounds_over_under": "pr",
    "player_points_assists_over_under": "pa",
    "player_rebounds_assists_over_under": "ra",
    "player_pts+rebs+asts": "pra",
}