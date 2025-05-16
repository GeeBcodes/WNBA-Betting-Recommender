import sys
import os

# Add project root to sys.path to allow for absolute imports from 'backend'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Example preprocessor
import joblib
import logging
from pathlib import Path
import datetime
import uuid # Added for UUIDs
from typing import Optional

# Database related imports
from sqlalchemy.orm import Session
from backend.db.session import SessionLocal # To create DB sessions
from backend.app import crud # To use CRUD functions
from backend.schemas import model_version as model_version_schema # For creating ModelVersion records

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure model artifact directory exists
MODEL_NAME_PREFIX = "wnba_player_prop_model"

# Determine current year for data path
CURRENT_YEAR = datetime.datetime.now().year
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / f"processed_player_data_{CURRENT_YEAR}.csv"

# Utility to get a database session
def get_db_session() -> Session:
    db = SessionLocal()
    try:
        # This is a context manager pattern if needed `yield` for a block
        # For simple calls, just returning db and closing in finally is also an option
        # However, for consistency with FastAPI `Depends`, yielding is fine if we manage its scope.
        # For a script, it might be simpler to create, use, and close in each function or pass around. 
        # For now, this function will just return a session that the caller must close.
        return db 
    except Exception as e:
        logging.error(f"Error creating database session: {e}")
        if db:
            db.close() # Ensure db is closed on error during creation if partially successful
        raise

def load_data(data_path: Path) -> pd.DataFrame:
    """Loads data from the specified path."""
    logging.info(f"Loading data from {data_path}")
    if not data_path.exists():
        logging.error(f"Data file not found at {data_path}")
        # For now, creating a dummy DataFrame for demonstration
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Example binary target
        })
    return pd.read_csv(data_path)

def train_model(df: pd.DataFrame):
    """Trains a model on the given DataFrame."""
    logging.info("Starting model training...")
    
    if df.empty or 'target' not in df.columns:
        logging.error("DataFrame is empty or 'target' column is missing. Skipping training.")
        return None

    # Define numeric features to be used for training
    # Exclude IDs like player_id if they are not intended as direct features for this model type
    # 'season' can be numeric but its utility as a raw feature depends on the model context.
    numeric_features = [
        'games_played', 'points', 'rebounds', 'assists', 
        'steals', 'blocks', 'ppg' # 'season' and 'player_id' could be added if appropriate
    ]
    
    # Ensure all selected numeric features actually exist in the DataFrame
    available_numeric_features = [col for col in numeric_features if col in df.columns]
    if not available_numeric_features:
        logging.error("No numeric features available for training after selection. Skipping training.")
        return None
    
    logging.info(f"Using the following numeric features for training: {available_numeric_features}")

    X = df[available_numeric_features] # Select only numeric features
    y = df['target']

    # Basic type checking for features - adapt as needed
    # This check might be redundant now, but can be a safeguard
    if not all(X[col].dtype in ['int64', 'float64'] for col in X.columns):
        logging.warning("Some features are not numeric. Model training might fail or be suboptimal.")
        # Add more sophisticated feature engineering/preprocessing here

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a scikit-learn pipeline
    # This is a very basic example. Real pipelines would include more feature engineering,
    # ColumnTransformer for different data types, etc.
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Example: scale numeric features
        ('classifier', RandomForestClassifier(random_state=42)) # Example classifier
    ])

    logging.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate model
    predictions = pipeline.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    logging.info(f"Model accuracy on test set: {acc:.4f}")

    return pipeline, acc # Return accuracy along with the pipeline

def save_model(pipeline, db: Session, model_dir: Path, model_name_prefix: str, accuracy: Optional[float] = None):
    """Saves the trained pipeline to a versioned file and creates a ModelVersion record in the DB."""
    if pipeline is None:
        logging.warning("No model to save.")
        return None # Return None if no model was saved

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name_prefix}_{timestamp}.joblib"
    model_path = model_dir / model_filename
    
    logging.info(f"Saving model artifact to {model_path}")
    joblib.dump(pipeline, model_path)
    logging.info("Model artifact saved successfully.")

    # Create ModelVersion entry in the database
    model_version_data = model_version_schema.ModelVersionCreate(
        version_name=model_filename, # Use the filename as the version name
        description=f"RandomForestClassifier trained on {timestamp}. Accuracy: {accuracy if accuracy is not None else 'N/A'}. Path: {model_path}"
    )
    try:
        db_model_version = crud.create_model_version(db=db, model_version=model_version_data)
        logging.info(f"Successfully created ModelVersion record with ID: {db_model_version.id} and Name: {db_model_version.version_name}")
        return db_model_version # Return the created DB object
    except Exception as e:
        logging.error(f"Failed to create ModelVersion record in DB: {e}")
        # Potentially consider what to do if file is saved but DB record fails
        return None

def main():
    """Main function to orchestrate data loading, training, and saving."""
    logging.info("Starting the model training pipeline...")
    
    db: Optional[Session] = None # Initialize db session variable
    try:
        db = get_db_session()
        # 1. Load data
        df = load_data(DATA_PATH)
        
        # 2. Train model
        # Ensure train_model can handle df being potentially empty from load_data error state
        if df.empty:
            logging.error("Data loading failed or returned empty data. Skipping model training and saving.")
            return # Exit main if no data
            
        trained_pipeline_results = train_model(df)
        
        if trained_pipeline_results is None:
            logging.warning("Model training did not produce a pipeline. Nothing to save.")
            return # Exit main if no pipeline trained

        trained_pipeline, accuracy = trained_pipeline_results
        
        # 3. Save model (and ModelVersion record)
        if trained_pipeline:
            saved_model_version = save_model(trained_pipeline, db, MODEL_DIR, MODEL_NAME_PREFIX, accuracy)
            if saved_model_version:
                logging.info(f"Model artifact and DB record saved successfully. Version: {saved_model_version.version_name}, Accuracy: {accuracy:.4f}")
            else:
                logging.warning("Model artifact saved, but failed to create ModelVersion DB record.")
            
    except Exception as e:
        logging.error(f"An error occurred in the main training pipeline: {e}", exc_info=True)
    finally:
        if db is not None:
            logging.info("Closing database session.")
            db.close()
        
    logging.info("Model training pipeline finished.")

if __name__ == "__main__":
    main() 