from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# In a real application, get this from environment variables
DATABASE_URL = "postgresql://postgres:Gr4t3fu143v3R@localhost:5432/wnba_db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
