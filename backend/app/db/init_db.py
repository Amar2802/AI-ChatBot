import json
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.database import engine, Base
from app.core.config import settings
from app.models.faq import FAQ

logger = logging.getLogger(__name__)

def init_db(db: Session):
    """
    Creates all database tables and seeds them with sample FAQs if empty.
    Configures WAL mode for SQLite.
    """
    # 1. Create tables
    Base.metadata.create_all(bind=engine)
    
    # 2. Configure SQLite WAL journal mode for concurrent read/write support
    try:
        db.execute(text("PRAGMA journal_mode=WAL;"))
        db.commit()
    except Exception as e:
        logger.warning(f"Could not enable WAL mode: {e}")
    
    # 3. Seed FAQs if database is empty
    try:
        if db.query(FAQ).count() == 0:
            if settings.FAQ_PATH.exists():
                logger.info(f"Seeding FAQs from {settings.FAQ_PATH}...")
                with open(settings.FAQ_PATH, "r", encoding="utf-8") as f:
                    items = json.load(f)
                
                faqs = [
                    FAQ(question=item["question"], answer=item["answer"])
                    for item in items
                ]
                db.bulk_save_objects(faqs)
                db.commit()
                logger.info(f"Successfully seeded {len(faqs)} FAQs.")
            else:
                logger.warning(f"FAQ JSON file not found at {settings.FAQ_PATH}. Skipping seeding.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding FAQs: {e}")
