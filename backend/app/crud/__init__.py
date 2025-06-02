# Import and expose all asynchronous CRUD functions

from .games import (
    create_game,
    get_game,
    get_game_by_external_id,
    get_games,
)

from .model_versions import (
    create_model_version,
    get_model_version,
    get_model_version_by_name,
    get_model_versions,
)

from .odds import (
    get_game_odd,
    get_game_odds,
    get_player_prop,
    get_player_props,
)

from .parlays import (
    create_parlay,
    get_parlay,
    get_parlays,
)

from .players import (
    create_player,
    get_player,
    get_player_by_api_id,
    get_players,
)

from .player_stats import (
    get_player_stat,
    get_player_stats,
    # Add other async player_stat CRUD functions here if they are created, e.g.:
    # async_create_player_stat,
    # async_update_player_stat,
    # async_delete_player_stat,
)

from .predictions import (
    create_prediction,
    get_prediction,
    get_predictions,
    get_predictions_by_player_prop,
    get_predictions_by_model_version,
)


__all__ = [
    # Games
    "create_game",
    "get_game",
    "get_game_by_external_id",
    "get_games",

    # Model Versions
    "create_model_version",
    "get_model_version",
    "get_model_version_by_name",
    "get_model_versions",

    # Odds
    "get_game_odd",
    "get_game_odds",
    "get_player_prop",
    "get_player_props",

    # Parlays
    "create_parlay",
    "get_parlay",
    "get_parlays",

    # Players
    "create_player",
    "get_player",
    "get_player_by_api_id",
    "get_players",

    # Player Stats
    "get_player_stat",
    "get_player_stats",
    # "async_create_player_stat", 
    # "async_update_player_stat",
    # "async_delete_player_stat",

    # Predictions
    "create_prediction",
    "get_prediction",
    "get_predictions",
    "get_predictions_by_player_prop",
    "get_predictions_by_model_version",
]

# Placeholder for player_stats CRUD functions if they are directly in this file or need specific handling
# For now, we assume player_stats.py router was directly calling functions that might need to be centralized here.
# If backend.app.crud was a Pylance-suggested auto-import that didn't point to a real module yet,
# then the actual player_stats crud functions (like crud.get_player_stats from player_stats router) need to be defined or moved here.

# For the player_stats router, it was using crud.get_player_stats directly.
# This implies that there should be a get_player_stats function available when importing `backend.app.crud`.
# Let's assume for now that these are defined in a separate file `crud/player_stats.py`
# and player_stats router correctly imports them.

# If player_stats crud functions are not in their own file, they need to be added here or in a separate player_stats.py under crud/

# For now, focusing on exposing the odds functions.
# The player_stats router uses `from backend.app import crud`
# and then calls `crud.get_player_stats(...)`
# This structure implies that `get_player_stats` should be available from this `__init__.py`

# Let's create a placeholder for where player_stats crud functions would be imported from.
# If they are not in a separate file, this init needs to define them or they should be moved.

# For simplicity and to ensure the odds router works, we primarily focus on exporting odds functions.
# The structure of player_stats crud needs to be confirmed if it's not already in crud/player_stats.py

# To make `from backend.app import crud` and then `crud.get_player_stats` work as in the player_stats router,
# we need to ensure `get_player_stats` is exposed.
# Let's assume there's a `crud.player_stats` module with `get_player_stats`.

try:
    from .player_stats import get_player_stats as get_player_stats_from_module # Add other player_stat cruds if they exist
    # Add to __all__ if found
    if 'get_player_stats_from_module' in locals():
        __all__.append("get_player_stats_from_module") 
        # You might want to alias it back to get_player_stats in __all__ for consistency if the router uses that name
        # e.g. __all__ = [..., "get_player_stats_from_module as get_player_stats"]
        # Or more simply, just ensure `get_player_stats` is directly available.
except ImportError:
    # This means backend.app.crud.player_stats does not exist or get_player_stats is not in it.
    # For now, the odds router doesn't depend on it, so we can proceed.
    # This will need to be resolved for the player_stats router to function correctly with this new structure.
    print("Warning: backend.app.crud.player_stats or its functions not found. Player stats API might be affected.")
    pass 