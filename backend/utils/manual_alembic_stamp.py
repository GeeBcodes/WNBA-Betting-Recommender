import sys
import os
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.db.session import SessionLocal
from sqlalchemy import text
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_REVISION = '93c142e1ee02'

def manually_stamp_revision():
    db: Session = SessionLocal()
    try:
        logger.info(f"Attempting to manually stamp Alembic revision to: {TARGET_REVISION}")

        # 1. Ensure alembic_version table exists
        create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        );
        """)
        db.execute(create_table_sql)
        logger.info("Ensured alembic_version table exists.")

        # 2. Delete any existing version numbers to avoid conflicts / ensure clean state
        delete_sql = text("DELETE FROM alembic_version;")
        db.execute(delete_sql)
        logger.info("Cleared existing records from alembic_version table.")

        # 3. Insert the target revision
        insert_sql = text("INSERT INTO alembic_version (version_num) VALUES (:version);")
        db.execute(insert_sql, {"version": TARGET_REVISION})
        logger.info(f"Successfully inserted revision {TARGET_REVISION} into alembic_version table.")
        
        db.commit()
        logger.info("Manual stamp committed.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during manual Alembic stamp: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    manually_stamp_revision() 