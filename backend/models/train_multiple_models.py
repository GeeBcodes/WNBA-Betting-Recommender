import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging
import datetime
import time

# Add project root to sys.path to allow for absolute imports
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.train_model import main as train_single_model_async
from backend.app.core.config import DEFAULT_SEASONS, TARGET_STATS_TO_TRAIN, TARGET_TRANSFORM_FOR_REGRESSION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_all_training_processes(seasons: list):
    """
    Runs the training process for all specified target stats and model types asynchronously.
    """
    total_models = len(TARGET_STATS_TO_TRAIN) * 2  # Regression and Classification for each
    logger.info(f"Starting consecutive training for {len(TARGET_STATS_TO_TRAIN)} target stats, generating {total_models} models.")
    
    successful_trainings = 0
    failed_trainings = []
    model_counter = 0

    for target_stat in TARGET_STATS_TO_TRAIN:
        for model_type in ['regression', 'classification']:
            model_counter += 1
            logger.info(f"--- ({model_counter}/{total_models}) Starting training for: '{target_stat}' ({model_type}) ---")
            
            start_time = time.time()
            
            try:
                # Determine if transformation should be applied for this regression model
                transform_needed = (
                    model_type == 'regression' and 
                    TARGET_TRANSFORM_FOR_REGRESSION.get(target_stat, False)
                )

                # Call the main async training function from train_model.py
                await train_single_model_async(
                    target_stat=target_stat,
                    model_type=model_type,
                    seasons=seasons,
                    transform=transform_needed
                )
                
                end_time = time.time()
                duration = end_time - start_time

                logger.info(f"--- ({model_counter}/{total_models}) COMPLETED training for '{target_stat}' ({model_type}) in {duration:.2f} seconds. ---")
                successful_trainings += 1

            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.critical(f"--- ({model_counter}/{total_models}) CRITICAL ERROR during training for '{target_stat}' ({model_type}) after {duration:.2f} seconds. Error: {e}")
                # We re-raise the exception's traceback in the log
                logger.critical(e, exc_info=True)
                failed_trainings.append(f"{target_stat} ({model_type})")

    logger.info("\n" + "="*50)
    logger.info("           Model Training Summary")
    logger.info("="*50)
    logger.info(f"Total models attempted: {total_models}")
    logger.info(f"Successful trainings: {successful_trainings}")
    logger.info(f"Failed trainings: {len(failed_trainings)}")
    if failed_trainings:
        logger.warning(f"Failed to train the following models: {', '.join(failed_trainings)}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all model training processes.")
    parser.add_argument("--seasons", nargs='*', type=int, default=DEFAULT_SEASONS, help="List of seasons to use for training.")
    args = parser.parse_args()
    asyncio.run(run_all_training_processes(seasons=args.seasons))