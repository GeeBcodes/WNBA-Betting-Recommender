import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load .env file if it exists (for local development outside Docker)
load_dotenv()

# Default to the hardcoded local URL if DATABASE_URL env var is not set
# Docker Compose will set DATABASE_URL for the backend service.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Gr4t3fu143v3R@localhost:5432/wnba_db")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
