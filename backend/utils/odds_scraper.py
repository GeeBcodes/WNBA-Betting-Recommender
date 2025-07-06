import os
import logging
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import time
import sqlalchemy as sa
import argparse
from typing import Optional, List, Dict, Set
from sqlalchemy import func
from sqlalchemy.orm import Session 
import unicodedata
# import json # No longer needed, but keeping here in case of file state issues
import traceback

# --- Setup Project Path ---
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set up, we can import our modules
from backend.app.crud import teams as crud_teams
from backend.db.session import get_sync_db_session
from backend.db.models import Game, Player, PlayerProp, Team, Bookmaker, Market, Sport
from backend.schemas import game as game_schema

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment and API Configuration ---
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
WNBA_SPORT_KEY = "basketball_wnba"
REGIONS = "us,us_dfs"
API_DELAY = 3

# --- Data Definitions ---
BOOKMAKERS = {"prizepicks", "underdog", "bovada", "mybookieag"}
PLAYER_PROPS_MARKETS = {
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_blocks", "player_steals", "player_blocks_steals", "player_turnovers",
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
}

# --- Normalization Helpers ---
TEAM_NAME_MAP = {
    "Atlanta Dream": "Dream", "Chicago Sky": "Sky", "Connecticut Sun": "Sun",
    "Dallas Wings": "Wings", "Indiana Fever": "Fever", "Las Vegas Aces": "Aces",
    "Los Angeles Sparks": "Sparks", "Minnesota Lynx": "Lynx", "New York Liberty": "Liberty",
    "Phoenix Mercury": "Mercury", "Seattle Storm": "Storm", "Washington Mystics": "Mystics",
    "Golden State Valkyries": "Valkyries"
}

def normalize_player_name(name: str) -> str:
    """Aggressively normalizes a player's name for robust matching."""
    if not name: return ""
    # Normalize unicode characters (e.g., 'Azurá' -> 'Azura')
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    # Convert to lowercase, remove periods, and strip whitespace
    name = name.lower().replace('.', '').strip()
    # Manual override for specific known discrepancies
    if name == "skylar diggins-smith":
        name = "skylar diggins"
    return name

# --- API Request Helper ---
def make_api_request(url: str, params: dict) -> Optional[dict]:
    try:
        response = requests.get(url, params=params, timeout=30)
        remaining = response.headers.get('X-Requests-Remaining')
        if remaining:
            logger.info(f"API Quota Remaining: {remaining}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None

# --- Database Operations ---
def get_game(db: Session, event: dict) -> Optional[Game]:
    """Finds a game in the DB matching the API event data, creating it if necessary."""
    # Strategy 1: Find the game by its unique API ID. This is the most reliable method.
    api_game_id = event.get('id')
    if api_game_id:
        game = db.query(Game).filter(Game.the_odds_api_game_id == api_game_id).first()
        if game:
            logger.info(f"Successfully matched event to Game ID {game.id} using API game ID.")
            return game

    # --- Fallback to matching by teams and date if API ID lookup fails ---
    try:
        game_dt_utc = datetime.fromisoformat(event.get("commence_time").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        logger.error(f"Invalid or missing commence_time in event: {event.get('id')}")
        return None

    logger.info(f"--- Attempting to match event: {event.get('home_team')} vs {event.get('away_team')} at {game_dt_utc} by teams and date ---")

    home_team_name_from_api = event.get('home_team', '')
    away_team_name_from_api = event.get('away_team', '')

    home_team_db_name = TEAM_NAME_MAP.get(home_team_name_from_api)
    away_team_db_name = TEAM_NAME_MAP.get(away_team_name_from_api)

    if not home_team_db_name or not away_team_db_name:
        logger.warning(f"Could not normalize one or both team names from API: '{home_team_name_from_api}', '{away_team_name_from_api}'.")
        return None

    home_team = crud_teams.get_team_by_name(db, name=home_team_db_name)
    away_team = crud_teams.get_team_by_name(db, name=away_team_db_name)

    if not home_team or not away_team:
        logger.warning(f"Could not find one or both teams in the database by mapped name: {home_team_db_name}, {away_team_db_name}")
        return None

    api_date = game_dt_utc.date()
    logger.info(f"Searching for game between {home_team_db_name} and {away_team_db_name} on the UTC date: {api_date}")

    games_on_date = db.query(Game).filter(
        func.date(Game.game_datetime) == api_date,
        ((Game.home_team_id == home_team.id) & (Game.away_team_id == away_team.id)) |
        ((Game.home_team_id == away_team.id) & (Game.away_team_id == home_team.id))
    ).all()

    if len(games_on_date) == 1:
        game = games_on_date[0]
        logger.info(f"Successfully matched event to Game ID: {game.id}")
        return game
    elif len(games_on_date) > 1:
        logger.warning(f"Found multiple games ({len(games_on_date)}) on the same date. Selecting the one closest in time.")
        closest_game = min(games_on_date, key=lambda g: abs(g.game_datetime.replace(tzinfo=timezone.utc) - game_dt_utc))
        logger.info(f"Selected closest game. Matched event to Game ID: {closest_game.id}")
        return closest_game
    else:
        logger.warning(f"Could not find a matching game for event {api_game_id}. Creating a new one.")
        
        # Create the game. If this fails (e.g., because the lookup failed and the game
        # already exists), the exception will now correctly propagate up to the main
        # try/except block, which will perform a rollback on the entire session.
        new_game_schema = game_schema.GameCreate(
            the_odds_api_game_id=event['id'],
            game_datetime=game_dt_utc,
            season=game_dt_utc.year,
            status='Scheduled',
            home_team_id=home_team.id,
            away_team_id=away_team.id
        )
        
        new_game_data = {k: v for k, v in new_game_schema.model_dump().items() if v is not None}
        db_game = Game(**new_game_data)
        db.add(db_game)
        db.flush()  # Send the insert to the DB to check for errors immediately.
        db.refresh(db_game)
        logger.info(f"Successfully added new game with ID: {db_game.id} to session.")
        return db_game

def get_player(db: Session, player_name: str, team_ids: list[int]) -> Optional[Player]:
    """Finds a player in the DB matching the name, constrained by team. Tries multiple strategies."""
    normalized_name = normalize_player_name(player_name)
    
    # Strategy 1: Exact full name match (case-insensitive), as this is the most reliable.
    player = db.query(Player).filter(
        func.lower(Player.player_name) == normalized_name,
        Player.team_id.in_(team_ids)
    ).first()
    if player:
        return player

    # Fallback Strategy 2: Contains match (case-insensitive) to handle partial names or slight differences.
    player = db.query(Player).filter(
        func.lower(Player.player_name).contains(normalized_name),
        Player.team_id.in_(team_ids)
    ).first()
    if player:
        return player

    return None

def process_odds_data(db: Session, event_odds: dict):
    """Processes and stores player prop odds for a single event."""
    game = get_game(db, event_odds)
    if not game:
        logger.warning(f"Skipping odds for event ID {event_odds['id']} as no matching game was found.")
        return

    # Pre-fetch bookmaker and market IDs for efficiency
    bookmaker_key_to_id = {b.key: b.id for b in db.query(Bookmaker).filter(Bookmaker.key.in_(BOOKMAKERS)).all()}
    market_key_to_id = {m.key: m.id for m in db.query(Market).filter(Market.key.in_(PLAYER_PROPS_MARKETS)).all()}
    
    home_team_id = game.home_team_id
    away_team_id = game.away_team_id
    
    all_props_to_upsert = []

    for bookmaker_data in event_odds.get("bookmakers", []):
        bookmaker_key = bookmaker_data.get("key")
        bookmaker_id = bookmaker_key_to_id.get(bookmaker_key)
        if not bookmaker_id:
            continue # Skip bookmakers we don't care about

        for market_data in bookmaker_data.get("markets", []):
            market_key = market_data.get("key")
            market_id = market_key_to_id.get(market_key)
            if not market_id:
                continue # Skip markets we don't care about

            # This dictionary will hold {('Player Name', 10.5): {'over': 1.80, 'under': 1.90}}
            player_lines = {}
            for outcome in market_data.get("outcomes", []):
                player_name_api = outcome.get("description")
                line = outcome.get("point")

                if not player_name_api or line is None:
                    continue

                prop_key = (player_name_api, line)
                if prop_key not in player_lines:
                    player_lines[prop_key] = {}
                
                outcome_name = outcome.get("name", "").lower()
                if outcome_name == "over":
                    player_lines[prop_key]["over_price"] = outcome.get("price")
                elif outcome_name == "under":
                    player_lines[prop_key]["under_price"] = outcome.get("price")

            # Now that we've grouped over/under pairs, process them
            for (player_name_api, line), prices in player_lines.items():
                # Crucially, search for the player ONLY on the two teams in this game
                player = get_player(db, player_name_api, [home_team_id, away_team_id])
                
                if not player:
                    # This will correctly log players who are in the API but not on the competing teams, or can't be matched
                    logger.warning(f"Could not match player '{player_name_api}' to a player on the competing teams for game {game.id}. Skipping prop.")
                    continue
                    
                # Found a valid player, prepare the prop data for upsert
                
                # Construct the 'outcomes' JSON field
                # In the future, this could be expanded to include alternate lines if the API provides them
                outcomes_json = [
                    {
                        "name": player_name_api,
                        "price": prices.get("over_price"),
                        "point": line,
                        "line_id": "standard_over" # Or some identifier
                    },
                    {
                        "name": player_name_api,
                        "price": prices.get("under_price"),
                        "point": line,
                        "line_id": "standard_under"
                    }
                ]

                prop_data = {
                    "game_id": game.id,
                    "player_id": player.id,
                    "market_id": market_id,
                    "bookmaker_id": bookmaker_id,
                    "line": line,
                    "over_price": prices.get("over_price"),
                    "under_price": prices.get("under_price"),
                    "last_update_api": datetime.fromisoformat(market_data["last_update"].replace("Z", "+00:00")),
                    "outcomes": outcomes_json
                }
                all_props_to_upsert.append(prop_data)

    if all_props_to_upsert:
        logger.info(f"Identified {len(all_props_to_upsert)} valid player props to upsert for game {game.id}")
        for prop_data in all_props_to_upsert:
                        existing_prop = db.query(PlayerProp).filter_by(
                game_id=prop_data['game_id'],
                player_id=prop_data['player_id'],
                market_id=prop_data['market_id'],
                bookmaker_id=prop_data['bookmaker_id'],
                line=prop_data['line']
                        ).first()

                        if existing_prop:
                             # Update existing prop
                            for key, value in prop_data.items():
                                setattr(existing_prop, key, value)
                        else:
                             # Insert new prop
                            new_prop = PlayerProp(**prop_data)
                            db.add(new_prop)
    else:
        logger.info(f"No valid, matchable player props found for game {game.id} from the API response.")
    


# --- Main Orchestration ---
def fetch_and_store_player_props(db: Session):
    """
    Fetches upcoming events and their player prop odds from the API,
    then processes and stores them in the database.
    """
    if not THE_ODDS_API_KEY:
        logger.error("THE_ODDS_API_KEY environment variable not set. Exiting.")
        return

    events_url = f"{API_BASE_URL}/{WNBA_SPORT_KEY}/events"
    events_params = {"apiKey": THE_ODDS_API_KEY}
    events = make_api_request(events_url, events_params)

    if not events:
        logger.error("Failed to fetch WNBA events. Exiting.")
        return

    logger.info(f"Found {len(events)} upcoming events.")

    for i, event in enumerate(events):
        logger.info(f"Fetching odds for event: {event['home_team']} vs {event['away_team']}")
        odds_url = f"{API_BASE_URL}/{WNBA_SPORT_KEY}/events/{event['id']}/odds"
        markets_to_fetch = ",".join(PLAYER_PROPS_MARKETS)
        odds_params = {
            "apiKey": THE_ODDS_API_KEY, "regions": REGIONS,
            "markets": markets_to_fetch
        }
        event_odds = make_api_request(odds_url, odds_params)
        
        if event_odds:
            process_odds_data(db, event_odds)
    else:
            logger.warning(f"Could not retrieve odds for event ID {event['id']}")
        
    if i < len(events) - 1:
            time.sleep(API_DELAY)

def main():
    logger.info("Starting WNBA player props scraper.")
    with get_sync_db_session() as db:
        try:
            # The default action is to fetch current player props for upcoming games
            fetch_and_store_player_props(db)
            # The commit is now handled by the session context manager
            logger.info("Database operations completed successfully and committed.")
        except Exception as e:
            # The rollback is also handled by the session context manager
            logger.error(f"An error occurred during the scraping process, changes have been rolled back: {e}")
            logger.error(traceback.format_exc())
    logger.info("Scraping finished.")

if __name__ == "__main__":
    main() 