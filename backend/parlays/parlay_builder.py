import sys
import os
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session, joinedload, selectinload

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import SessionLocal
from backend.db import models as db_models
from backend.schemas import parlay as parlay_schema
from backend.schemas import prediction as prediction_schema # For accessing prediction details
from backend.app.crud import predictions as crud_predictions # To fetch predictions
from backend.app.crud import parlays as crud_parlays # To save parlays

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Parlay Generation ---
MIN_PROBABILITY_THRESHOLD = 0.55  # Temporarily lowered from 0.60 for testing
MAX_LEGS_PER_PARLAY = 3          # e.g., limit parlays to 2-3 legs for reasonable probability
MIN_LEGS_PER_PARLAY = 2
# Consider adding Kelly Criterion or similar for bet sizing/edge calculation later if desired

def get_potential_parlay_legs(db: Session, game_date: Optional[datetime.date] = None) -> List[prediction_schema.Prediction]:
    """
    Fetches predictions that meet the criteria for being a parlay leg.
    """
    logger.info(f"Fetching potential parlay legs with probability >= {MIN_PROBABILITY_THRESHOLD}")
    
    # Use the existing CRUD function to get predictions, then filter them in Python
    # This is simpler than complex SQL filters for now, can be optimized if performance is an issue.
    all_predictions = crud_predictions.get_predictions(db, limit=1000, game_date=game_date) # Get a large batch
    
    potential_legs: List[prediction_schema.Prediction] = []
    for pred_orm_object in all_predictions:
        # Convert ORM object to Pydantic schema to easily access attributes and nested data
        # The crud_predictions.get_predictions already returns ORM objects with eager loaded relationships
        # So, we can directly use pred_orm_object attributes
        
        if pred_orm_object.predicted_over_probability and pred_orm_object.predicted_over_probability >= MIN_PROBABILITY_THRESHOLD:
            # This is a potential "over" leg
            # We might want to store which side (over/under) is the pick
            potential_legs.append(prediction_schema.Prediction.model_validate(pred_orm_object))
        elif pred_orm_object.predicted_under_probability and pred_orm_object.predicted_under_probability >= MIN_PROBABILITY_THRESHOLD:
            # This is a potential "under" leg
            potential_legs.append(prediction_schema.Prediction.model_validate(pred_orm_object))
            
    logger.info(f"Found {len(potential_legs)} potential parlay legs.")
    return potential_legs


def calculate_parlay_details(legs: List[prediction_schema.Prediction]) -> Optional[Dict[str, Any]]:
    """
    Calculates combined probability and odds for a given set of parlay legs.
    Assumes independence of legs for probability calculation.
    """
    if not legs or len(legs) < MIN_LEGS_PER_PARLAY:
        logger.warning("Not enough legs to form a parlay.")
        return None

    combined_probability = 1.0
    selections_details = []

    for leg_prediction in legs:
        chosen_probability = 0.0
        chosen_outcome_description = ""
        # Determine if we are betting on 'Over' or 'Under' based on which probability met the threshold
        # This logic assumes get_potential_parlay_legs added it based on one side meeting threshold
        if leg_prediction.predicted_over_probability and leg_prediction.predicted_over_probability >= MIN_PROBABILITY_THRESHOLD:
            chosen_probability = leg_prediction.predicted_over_probability
            chosen_outcome_description = "Over"
        elif leg_prediction.predicted_under_probability and leg_prediction.predicted_under_probability >= MIN_PROBABILITY_THRESHOLD:
            chosen_probability = leg_prediction.predicted_under_probability
            chosen_outcome_description = "Under"
        else:
            # This case should ideally not happen if legs are sourced from get_potential_parlay_legs
            logger.error(f"Prediction ID {leg_prediction.id} has no probability meeting threshold. Skipping.")
            continue
        
        if chosen_probability <= 0: # Avoid issues with log(0) or division by zero if converting to odds
            logger.warning(f"Skipping leg {leg_prediction.id} due to non-positive probability ({chosen_probability})")
            return None # Or handle more gracefully, e.g. by excluding the leg

        combined_probability *= chosen_probability
        
        # Extract line/point for the chosen outcome. Need to access PlayerProp outcomes.
        line_point = None
        # The leg_prediction.player_prop is already a Pydantic model (PlayerPropRead)
        if leg_prediction.player_prop and leg_prediction.player_prop.outcomes:
            # Find the outcome that matches our bet (Over/Under)
            # The structure of outcomes is typically like: {"name": "Player Name", "description": "Over", "price": -110, "point": 15.5}
            # or {"name": "Over", "price": -110, "point": 15.5}
            # We need to be robust here.
            for outcome_detail in leg_prediction.player_prop.outcomes:
                # Simple check for 'Over' or 'Under' in description or name
                if (chosen_outcome_description.lower() in outcome_detail.get('description','').lower() or \
                    chosen_outcome_description.lower() in outcome_detail.get('name','').lower()):
                    line_point = outcome_detail.get('point')
                    break
        
        selections_details.append({
            "prediction_id": str(leg_prediction.id),
            "player_prop_id": str(leg_prediction.player_prop_id),
            "player_name": leg_prediction.player_prop.player.player_name if leg_prediction.player_prop and leg_prediction.player_prop.player else "N/A",
            "market_key": leg_prediction.player_prop.market.key if leg_prediction.player_prop and leg_prediction.player_prop.market else "N/A",
            "game_id": str(leg_prediction.player_prop.game_id) if leg_prediction.player_prop else "N/A",
            "line_point": line_point,
            "chosen_outcome": chosen_outcome_description, # Over or Under
            "chosen_probability": chosen_probability
        })

    if not selections_details or len(selections_details) < MIN_LEGS_PER_PARLAY:
        logger.warning("Not enough valid legs after processing to form a parlay.")
        return None
    
    # Calculate total odds (Decimal odds from probability)
    # Avoid division by zero if combined_probability is 0
    total_decimal_odds = (1 / combined_probability) if combined_probability > 0 else float('inf') 

    return {
        "selections": selections_details,
        "combined_probability": combined_probability,
        "total_decimal_odds": total_decimal_odds
    }

# TODO: Function to generate combinations of parlays from potential legs
# TODO: Function to save a parlay to the DB using crud_parlays.create_parlay

def generate_and_store_parlays(db: Session, game_date: Optional[datetime.date] = None):
    """Main function to generate and store parlays."""
    # 1. Get potential legs
    potential_legs = get_potential_parlay_legs(db, game_date=game_date)
    
    if len(potential_legs) < MIN_LEGS_PER_PARLAY:
        logger.info("Not enough potential legs to generate parlays.")
        return

    # 2. Generate combinations of parlays (e.g., all 2-leg and 3-leg parlays)
    # This can be complex. For a start, let's try to make one parlay with the top N legs.
    # For a more robust solution, one would use itertools.combinations.
    
    # Sort legs by probability (descending) to pick the 'best' ones for a simple parlay
    # This requires knowing which probability (over/under) made it a potential leg.
    # Let's refine this: the `potential_legs` list now contains Prediction schemas.
    # When we calculated them, we chose either over or under probability.
    # We need a way to sort based on that chosen probability.
    
    # For simplicity for now, let's assume we just take the first MAX_LEGS_PER_PARLAY
    # from the list, assuming they are already somewhat ordered or we apply a simple sort.
    # A better approach is to store the 'chosen_probability' and 'chosen_side' with the leg.
    
    # Simplified approach for now: take the top N legs based on their highest probability (over or under)
    # This isn't perfect as it doesn't distinguish the side, but get_potential_parlay_legs already did.
    
    # Let's just try one parlay with the first MAX_LEGS_PER_PARLAY that meet the criteria
    # The get_potential_parlay_legs already filters by threshold.
    
    from itertools import combinations

    # Generate parlays for MIN_LEGS_PER_PARLAY up to MAX_LEGS_PER_PARLAY
    parlays_created_count = 0
    for num_legs_in_this_parlay in range(MIN_LEGS_PER_PARLAY, MAX_LEGS_PER_PARLAY + 1):
        if len(potential_legs) < num_legs_in_this_parlay:
            continue # Not enough legs for this size of parlay

        for leg_combination_pydantic in combinations(potential_legs, num_legs_in_this_parlay):
            # leg_combination_pydantic is a tuple of prediction_schema.Prediction objects
            parlay_details = calculate_parlay_details(list(leg_combination_pydantic))
            
            if parlay_details:
                logger.info(f"Calculated parlay: Prob={parlay_details['combined_probability']:.4f}, Legs={len(parlay_details['selections'])}")
                # Store this parlay
                parlay_create_data = parlay_schema.ParlayCreate(
                    selections=parlay_details["selections"], # Store detailed selections
                    combined_probability=parlay_details["combined_probability"],
                    total_odds=parlay_details["total_decimal_odds"] # Assuming we store decimal odds
                )
                try:
                    crud_parlays.create_parlay(db=db, parlay=parlay_create_data)
                    parlays_created_count +=1
                except Exception as e:
                    logger.error(f"Error saving parlay: {e}", exc_info=True)
                    # Continue to next parlay combination
    
    logger.info(f"Finished generating parlays. Created {parlays_created_count} parlays.")

if __name__ == "__main__":
    db_session = SessionLocal()
    try:
        # Ensure CRUD for parlays exists and is imported
        if not hasattr(crud_parlays, 'create_parlay'):
            logger.error("crud_parlays.create_parlay function not found. Please implement it.")
        else:
            # For testing, run for today's date (or a specific date with predictions)
            generate_and_store_parlays(db_session, game_date=datetime.today().date())
            # Example with a specific date, useful if you have historical predictions:
            # test_date = datetime.strptime("2024-05-16", "%Y-%m-%d").date()
            # generate_and_store_parlays(db_session, game_date=test_date)
    finally:
        db_session.close() 