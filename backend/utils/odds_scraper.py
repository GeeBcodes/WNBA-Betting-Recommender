import os
import logging
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import time # For rate limiting
import sqlalchemy as sa # For functions like sa.func.date and sa.func.lower

# Adjust sys.path to include the project root if this script is run directly
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.db.session import SessionLocal # For direct script execution
# from backend.app.dependencies import get_db # Not used directly in this script execution flow
from backend.db import models # Access models like models.Sport
from backend.schemas import odds as odds_schemas # Use alias to avoid name clashes
# Specific model imports for clarity in function type hints and queries
from backend.db.models import Sport, Bookmaker, Market, Game, Player, GameOdd, PlayerProp 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env file from the project root
dotenv_path = os.path.join(project_root, '.env')
if not os.path.exists(dotenv_path):
    logging.warning(f".env file not found at {dotenv_path}. ODDS_API_KEY must be set via environment variables.")
load_dotenv(dotenv_path=dotenv_path)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    logging.error("ODDS_API_KEY not found. Please set it in .env file or as an environment variable.")
    # Depending on desired behavior, you might exit or raise an error here.

# --- The Odds API Configuration ---
API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
WNBA_SPORT_KEY = "basketball_wnba"
REGIONS = ["us"] 

BOOKMAKERS_OF_INTEREST = {
    "bovada": "Bovada",
    "mybookieag": "MyBookie.ag",
    "prizepicks": "PrizePicks",
    "underdog": "Underdog Fantasy"
}

GAME_MARKETS = {
    "h2h": "Head to Head (Moneyline)",
    "spreads": "Point Spread (Handicap)",
    "totals": "Total Points (Over/Under)"
}

PLAYER_PROPS_MARKETS = {
    "player_points": "Player Points (Over/Under)",
    "player_points_q1": "Player 1st Quarter Points (Over/Under)",
    "player_rebounds": "Player Rebounds (Over/Under)",
    "player_rebounds_q1": "Player 1st Quarter Rebounds (Over/Under)",
    "player_assists": "Player Assists (Over/Under)",
    "player_assists_q1": "Player 1st Quarter Assists (Over/Under)",
    "player_threes": "Player Threes (Over/Under)",
    "player_blocks": "Player Blocks (Over/Under)",
    "player_steals": "Player Steals (Over/Under)",
    "player_blocks_steals": "Player Blocks + Steals (Over/Under)",
    "player_turnovers": "Player Turnovers (Over/Under)",
    "player_points_rebounds_assists": "Player Points + Rebounds + Assists (Over/Under)",
    "player_points_rebounds": "Player Points + Rebounds (Over/Under)",
    "player_points_assists": "Player Points + Assists (Over/Under)",
    "player_rebounds_assists": "Player Rebounds + Assists (Over/Under)",
    # The following might need specific checking if keys are exact on The Odds API
    "player_field_goals": "Player Field Goals (Over/Under)", 
    "player_alternate_points": "Alternate Player Points", # API uses player_points_alternate
    "player_alternate_rebounds": "Alternate Player Rebounds", # API uses player_rebounds_alternate
    "player_alternate_assists": "Alternate Player Assists", # API uses player_assists_alternate
    "player_alternate_threes": "Alternate Player Threes", # API uses player_threes_alternate
}
# Corrected Alternate Market Keys based on typical API patterns
# User provided list was extensive, ensure these keys match API reality.
# Focusing on a subset that is more commonly standardized or explicitly listed by The Odds API.
# The user list included items like 'player_frees_made', 'player_first_basket' etc.
# These often have very specific market keys or might not be universally available.
# Refer to The Odds API for exact player prop market keys beyond common ones.

# For now, let's use the keys as provided by user, assuming they are correct:
PLAYER_PROPS_MARKETS_FROM_USER = {
    "player_points": "Points (Over/Under)",
    "player_points_q1": "1st Quarter Points (Over/Under)",
    "player_rebounds": "Rebounds (Over/Under)",
    "player_rebounds_q1": "1st Quarter Rebounds (Over/Under)",
    "player_assists": "Assists (Over/Under)",
    "player_assists_q1": "1st Quarter Assists (Over/Under)",
    "player_threes": "Threes (Over/Under)",
    "player_blocks": "Blocks (Over/Under)",
    "player_steals": "Steals (Over/Under)",
    "player_blocks_steals": "Blocks + Steals (Over/Under)",
    "player_turnovers": "Turnovers (Over/Under)",
    "player_points_rebounds_assists": "Points + Rebounds + Assists (Over/Under)",
    "player_points_rebounds": "Points + Rebounds (Over/Under)",
    "player_points_assists": "Points + Assists (Over/Under)",
    "player_rebounds_assists": "Rebounds + Assists (Over/Under)",
    "player_field_goals": "Field Goals (Over/Under)",
    "player_frees_made": "Frees made (Over/Under)",
    "player_frees_attempts": "Frees attempted (Over/Under)",
    "player_first_basket": "First Basket Scorer (Yes/No)",
    "player_first_team_basket": "First Basket Scorer on Team (Yes/No)",
    "player_double_double": "Double Double (Yes/No)",
    "player_triple_double": "Triple Double (Yes/No)",
    "player_method_of_first_basket": "Method of First Basket (Various)",
    "player_points_alternate": "Alternate Points (Over/Under)",
    "player_rebounds_alternate": "Alternate Rebounds (Over/Under)",
    "player_assists_alternate": "Alternate Assists (Over/Under)",
    "player_blocks_alternate": "Alternate Blocks (Over/Under)",
    "player_steals_alternate": "Alternate Steals (Over/Under)",
    "player_turnovers_alternate": "Alternate Turnovers (Over/Under)",
    "player_threes_alternate": "Alternate Threes (Over/Under)",
    "player_points_assists_alternate": "Alternate Points + Assists (Over/Under)",
    "player_points_rebounds_alternate": "Alternate Points + Rebounds (Over/Under)",
    "player_rebounds_assists_alternate": "Alternate Rebounds + Assists (Over/Under)",
    "player_points_rebounds_assists_alternate": "Alternate Points + Rebounds + Assists (Over/Under)"
}
ALL_MARKETS = {**GAME_MARKETS, **PLAYER_PROPS_MARKETS_FROM_USER}

# --- Database Helper Functions ---
def _get_or_create_sport(db: Session, sport_key: str, sport_details: dict) -> Sport:
    """Gets or creates a sport record in the database."""
    sport = db.query(Sport).filter(Sport.key == sport_key).first()
    if not sport:
        sport_data = odds_schemas.SportCreate(
            key=sport_key,
            group_name=sport_details.get("group"),
            title=sport_details.get("title"),
            description=sport_details.get("description"),
            active=sport_details.get("active", True),
            has_outrights=sport_details.get("has_outrights", False)
        )
        sport = Sport(**sport_data.model_dump())
        db.add(sport)
        try:
            db.commit()
            db.refresh(sport)
            logging.info(f"Created sport: {sport.title} (Key: {sport.key})")
        except IntegrityError:
            db.rollback()
            sport = db.query(Sport).filter(Sport.key == sport_key).first()
            logging.info(f"Sport '{sport_key}' already exists, retrieved.")
        except Exception as e:
            db.rollback()
            logging.error(f"Error creating/retrieving sport {sport_key}: {e}", exc_info=True)
            raise # Re-raise after logging to halt if critical
    return sport

def _get_or_create_bookmaker(db: Session, bookmaker_key: str, bookmaker_title: str) -> Bookmaker:
    """Gets or creates a bookmaker record in the database."""
    bookmaker = db.query(Bookmaker).filter(Bookmaker.key == bookmaker_key).first()
    if not bookmaker:
        bookmaker_data = odds_schemas.BookmakerCreate(key=bookmaker_key, title=bookmaker_title)
        bookmaker = Bookmaker(**bookmaker_data.model_dump())
        db.add(bookmaker)
        try:
            db.commit()
            db.refresh(bookmaker)
            logging.info(f"Created bookmaker: {bookmaker.title} (Key: {bookmaker.key})")
        except IntegrityError:
            db.rollback()
            bookmaker = db.query(Bookmaker).filter(Bookmaker.key == bookmaker_key).first()
            logging.info(f"Bookmaker '{bookmaker_key}' already exists, retrieved.")
        except Exception as e:
            db.rollback()
            logging.error(f"Error creating/retrieving bookmaker {bookmaker_key}: {e}", exc_info=True)
            raise
    return bookmaker

def _get_or_create_market(db: Session, market_key: str, market_description: str) -> Market:
    """Gets or creates a market record in the database."""
    market = db.query(Market).filter(Market.key == market_key).first()
    if not market:
        market_data = odds_schemas.MarketCreate(key=market_key, description=market_description)
        market = Market(**market_data.model_dump())
        db.add(market)
        try:
            db.commit()
            db.refresh(market)
            logging.info(f"Created market: {market.description} (Key: {market.key})")
        except IntegrityError:
            db.rollback()
            market = db.query(Market).filter(Market.key == market_key).first()
            logging.info(f"Market '{market_key}' already exists, retrieved.")
        except Exception as e:
            db.rollback()
            logging.error(f"Error creating/retrieving market {market_key}: {e}", exc_info=True)
            raise
    return market

def _get_game_by_details(db: Session, home_team_name: str, away_team_name: str, game_api_id: str, game_datetime_utc: datetime) -> Game | None:
    """
    Tries to find a game by API ID first, then by home team, away team, and game date.
    The Odds API game_datetime is already in UTC (ISO 8601 format).
    Assumes team names from Odds API match those in our DB (e.g., from WNBA stats scraper).
    """
    if game_api_id:
        game = db.query(Game).filter(Game.external_id == game_api_id).first()
        if game:
            logging.debug(f"Found game by external_id: {game_api_id} (DB ID: {game.id})")
            return game

    game_date = game_datetime_utc.date()
    
    # Try direct match first
    game = db.query(Game).filter(
        Game.home_team == home_team_name,
        Game.away_team == away_team_name,
        sa.func.date(Game.game_datetime) == game_date
    ).order_by(Game.game_datetime.desc()).first() # Prefer latest if multiple on same date somehow
    if game:
        logging.debug(f"Found game by teams & date: {home_team_name} vs {away_team_name} on {game_date} (DB ID: {game.id})")
        return game
    
    # Try swapped teams (sometimes APIs list home/away differently)
    game = db.query(Game).filter(
        Game.home_team == away_team_name, 
        Game.away_team == home_team_name, 
        sa.func.date(Game.game_datetime) == game_date
    ).order_by(Game.game_datetime.desc()).first()
    if game:
        logging.debug(f"Found game by SWAPPED teams & date: {away_team_name} vs {home_team_name} on {game_date} (DB ID: {game.id})")
        return game

    logging.warning(f"Game not found for API ID: '{game_api_id}', Teams: '{home_team_name}' vs '{away_team_name}', Date: {game_date}")
    return None


def _get_player_by_name(db: Session, player_name: str) -> Player | None:
    """
    Finds a player by name. Case-insensitive search.
    This might require more sophisticated matching (e.g., fuzzy matching, aliases) in production.
    """
    normalized_name = player_name.strip() # Basic normalization
    player = db.query(Player).filter(sa.func.lower(Player.player_name) == sa.func.lower(normalized_name)).first()
    
    if not player:
        logging.warning(f"Player '{normalized_name}' (original from API: '{player_name}') not found in DB.")
        # Potential enhancement: Try fuzzy matching or search by first/last name components if direct match fails.
    return player


# --- Main API Fetching and Storing Logic ---
def fetch_and_store_wnba_odds(db: Session, sport_key: str, regions_str: str, bookmakers_str: str, markets_str: str):
    """
    Fetches odds for the WNBA from The Odds API for specified markets and bookmakers,
    then stores them in the database.
    Simplified player prop handling for this version.
    """
    if not ODDS_API_KEY:
        logging.error("Cannot fetch odds: ODDS_API_KEY is not set.")
        return

    wnba_sport_details = {"group": "Basketball", "title": "WNBA", "description": "Women's National Basketball Association"}
    _get_or_create_sport(db, sport_key, wnba_sport_details)

    endpoint_url = f"{API_BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions_str,
        "markets": markets_str,
        "bookmakers": bookmakers_str,
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }

    logging.info(f"Fetching odds from: {endpoint_url} with params: markets='{markets_str}', bookmakers='{bookmakers_str}'")

    try:
        response = requests.get(endpoint_url, params=params, timeout=45) # Increased timeout slightly
        response.raise_for_status()
        
        remaining_requests = response.headers.get('X-Requests-Remaining')
        if remaining_requests:
            logging.info(f"API Requests Remaining: {remaining_requests}")

        events_data = response.json()
        if not events_data:
            logging.info(f"No game events returned from API for markets: {markets_str}.")
            return

        logging.info(f"Received {len(events_data)} game events for markets: {markets_str}.")
        
        items_to_commit = []

        for event in events_data:
            api_game_id = event.get("id")
            home_team_api = event.get("home_team")
            away_team_api = event.get("away_team")
            commence_time_str = event.get("commence_time")

            if not all([api_game_id, home_team_api, away_team_api, commence_time_str]):
                logging.warning(f"Event missing critical data (id, teams, or time), skipping: {event.get('id', 'Unknown ID')}")
                continue
            
            try:
                game_datetime_utc = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
            except ValueError as e:
                logging.error(f"Invalid date format for event {api_game_id}: {commence_time_str}. Error: {e}")
                continue

            db_game = _get_game_by_details(db, home_team_api, away_team_api, api_game_id, game_datetime_utc)
            if not db_game:
                logging.warning(f"DB Game not found for API event: {api_game_id}, {home_team_api} vs {away_team_api}. Skipping odds.")
                continue

            for bookie_data in event.get("bookmakers", []):
                bookmaker_key_api = bookie_data.get("key")
                if bookmaker_key_api not in BOOKMAKERS_OF_INTEREST:
                    continue
                db_bookmaker = _get_or_create_bookmaker(db, bookmaker_key_api, BOOKMAKERS_OF_INTEREST.get(bookmaker_key_api, bookie_data.get("title")))

                for market_data_api in bookie_data.get("markets", []):
                    market_key_api = market_data_api.get("key")
                    if market_key_api not in ALL_MARKETS:
                        continue
                    db_market = _get_or_create_market(db, market_key_api, ALL_MARKETS[market_key_api])
                    
                    market_last_update_utc = None
                    if market_data_api.get("last_update"):
                        try:
                            market_last_update_utc = datetime.fromisoformat(market_data_api.get("last_update").replace("Z", "+00:00"))
                        except ValueError:
                            market_last_update_utc = datetime.now(timezone.utc) # Fallback

                    api_outcomes = market_data_api.get("outcomes")
                    if not api_outcomes:
                        continue

                    if market_key_api in GAME_MARKETS:
                        existing_odd = db.query(GameOdd).filter_by(game_id=db_game.id, bookmaker_id=db_bookmaker.id, market_id=db_market.id).first()
                        if existing_odd:
                            if existing_odd.last_update_api and market_last_update_utc and market_last_update_utc <= existing_odd.last_update_api:
                                continue # Skip if not newer
                            existing_odd.last_update_api = market_last_update_utc
                            existing_odd.outcomes = api_outcomes
                            if existing_odd not in items_to_commit: items_to_commit.append(existing_odd)
                        else:
                            new_odd = GameOdd(
                                game_id=db_game.id, bookmaker_id=db_bookmaker.id, market_id=db_market.id,
                                last_update_api=market_last_update_utc, outcomes=api_outcomes
                            )
                            if new_odd not in items_to_commit: items_to_commit.append(new_odd)

                    elif market_key_api in PLAYER_PROPS_MARKETS_FROM_USER:
                        for outcome_line in api_outcomes: # Each outcome_line is a specific bet, e.g., Player A Points Over 10.5
                            player_name_api = outcome_line.get("description")
                            if not player_name_api:
                                logging.warning(f"Player prop outcome missing player name for market {market_key_api}. Data: {outcome_line}")
                                continue
                            db_player = _get_player_by_name(db, player_name_api)
                            if not db_player:
                                logging.warning(f"Player '{player_name_api}' for prop market '{market_key_api}' not found. Skipping line.")
                                continue
                            
                            # Simplified: One PlayerProp DB entry per unique line (player, market, bookie, specific outcome like name+point).
                            # This requires a more granular unique constraint on PlayerProp or careful update logic.
                            # The current PlayerProp unique constraint in model is: game_id, player_name_api, bookmaker_id, market_id.
                            # This means one PlayerProp record stores ALL lines for that player for that market_id.

                            existing_prop_for_player_market = db.query(PlayerProp).filter_by(
                                game_id=db_game.id, player_id=db_player.id,
                                bookmaker_id=db_bookmaker.id, market_id=db_market.id
                            ).first()

                            if existing_prop_for_player_market:
                                if existing_prop_for_player_market.last_update_api and market_last_update_utc and market_last_update_utc <= existing_prop_for_player_market.last_update_api:
                                    # Check if this specific line needs update within the existing outcomes array
                                    line_updated = False
                                    if isinstance(existing_prop_for_player_market.outcomes, list):
                                        for i, ex_line in enumerate(existing_prop_for_player_market.outcomes):
                                            if ex_line.get('name') == outcome_line.get('name') and ex_line.get('point') == outcome_line.get('point'):
                                                existing_prop_for_player_market.outcomes[i] = outcome_line
                                                line_updated = True
                                                break
                                        if not line_updated:
                                            existing_prop_for_player_market.outcomes.append(outcome_line)
                                    else: # Overwrite if not a list or initialize
                                        existing_prop_for_player_market.outcomes = [outcome_line]
                                else: # Market data is newer, rebuild outcomes for this player prop
                                     existing_prop_for_player_market.outcomes = [outcome_line] # Start with this line, assuming subsequent lines for same player/market are appended in this pass
                                
                                existing_prop_for_player_market.last_update_api = market_last_update_utc
                                if existing_prop_for_player_market not in items_to_commit: items_to_commit.append(existing_prop_for_player_market)
                            
                            else: # Create new PlayerProp for this player for this market_id
                                new_prop = PlayerProp(
                                    game_id=db_game.id, player_id=db_player.id,
                                    bookmaker_id=db_bookmaker.id, market_id=db_market.id,
                                    player_name_api=player_name_api,
                                    last_update_api=market_last_update_utc,
                                    outcomes=[outcome_line] # Store the single line as a list with one item
                                )
                                if new_prop not in items_to_commit: items_to_commit.append(new_prop)
        if items_to_commit:
            try:
                db.add_all(items_to_commit)
                db.commit()
                logging.info(f"Committed {len(items_to_commit)} new/updated odds entries for markets: {markets_str}.")
            except Exception as e:
                db.rollback()
                logging.error(f"DB commit error for markets {markets_str}: {e}", exc_info=True)

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error for markets {markets_str}: {http_err} - Response: {http_err.response.text if http_err.response else 'No response'}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error for markets {markets_str}: {req_err}")
    except Exception as e:
        db.rollback()
        logging.error(f"Unexpected error for markets {markets_str}: {e}", exc_info=True)


def main():
    logging.info("Starting The Odds API ingestion pipeline...")
    db: Session = SessionLocal()
    try:
        regions_param = ",".join(REGIONS)
        bookies_param = ",".join(BOOKMAKERS_OF_INTEREST.keys())

        logging.info("--- Fetching Game Markets ---")
        game_markets_param = ",".join(GAME_MARKETS.keys())
        if game_markets_param:
            fetch_and_store_wnba_odds(db, WNBA_SPORT_KEY, regions_param, bookies_param, game_markets_param)
            time.sleep(2) # Pause between major API calls
        else:
            logging.info("No game markets configured to fetch.")

        logging.info("--- Fetching Player Prop Markets (one by one) ---")
        if not PLAYER_PROPS_MARKETS_FROM_USER:
            logging.info("No player prop markets configured to fetch.")
        else:
            for market_key in PLAYER_PROPS_MARKETS_FROM_USER.keys():
                logging.info(f"Fetching player prop market: {market_key}")
                fetch_and_store_wnba_odds(db, WNBA_SPORT_KEY, regions_param, bookies_param, market_key)
                time.sleep(5) # Be respectful of API limits, esp. free tier
    
    except Exception as e:
        logging.error(f"Critical error in main pipeline: {e}", exc_info=True)
    finally:
        db.close()
        logging.info("Odds ingestion pipeline finished.")


if __name__ == "__main__":
    logging.info("Odds API Scraper initialized with new imports and constants.")
    if ODDS_API_KEY:
        logging.info("ODDS_API_KEY loaded successfully.")
    else:
        logging.error("ODDS_API_KEY is missing. Cannot proceed with API calls.")
    logging.info(f"Targeting WNBA sport key: {WNBA_SPORT_KEY}")
    logging.info(f"Regions: {REGIONS}")
    logging.info(f"Bookmakers: {list(BOOKMAKERS_OF_INTEREST.keys())}")
    logging.info(f"Total markets configured: {len(ALL_MARKETS)}")

# Remove old placeholder functions and main logic from the original odds_scraper.py
# The old functions like fetch_game_odds, fetch_player_prop_odds, process_odds_data,
# and the old main() will be replaced by new logic using The Odds API. 