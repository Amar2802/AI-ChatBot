from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.session import Session as SessionModel
from app.models.message import Message as MessageModel
from app.models.feedback import Feedback as FeedbackModel
from app.models.faq import FAQ as FAQModel

def ensure_session(db: Session, session_id: str) -> SessionModel:
    """
    Ensure a session exists in the database. Inserts it if not.
    """
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        session = SessionModel(id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session

def add_message(db: Session, session_id: str, role: str, text: str) -> int:
    """
    Add a message to the database and return its ID.
    """
    ensure_session(db, session_id)
    message = MessageModel(session_id=session_id, role=role, text=text)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message.id

def get_recent_messages(db: Session, session_id: str, limit: int) -> List[Tuple[str, str]]:
    """
    Get recent message history (role, text) for a session.
    """
    messages = (
        db.query(MessageModel)
        .filter(MessageModel.session_id == session_id)
        .order_by(MessageModel.id.desc())
        .limit(limit * 2)
        .all()
    )
    # Reverse to restore chronological order
    messages.reverse()
    return [(m.role, m.text) for m in messages]

def log_feedback(db: Session, session_id: str, message_id: int, rating: int, note: Optional[str] = None):
    """
    Log user feedback for a message.
    """
    ensure_session(db, session_id)
    feedback = FeedbackModel(
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        note=note
    )
    db.add(feedback)
    db.commit()

def faq_count(db: Session) -> int:
    """
    Return count of FAQs in the database.
    """
    return db.query(FAQModel).count()

def insert_faqs(db: Session, pairs: List[Tuple[str, str]]):
    """
    Bulk insert FAQ pairs.
    """
    faqs = [FAQModel(question=q, answer=a) for q, a in pairs]
    db.bulk_save_objects(faqs)
    db.commit()

def replace_faqs(db: Session, pairs: List[Tuple[str, str]]):
    """
    Clear all FAQs and replace them.
    """
    db.query(FAQModel).delete()
    faqs = [FAQModel(question=q, answer=a) for q, a in pairs]
    db.bulk_save_objects(faqs)
    db.commit()

def dedupe_faqs(db: Session):
    """
    Deduplicate FAQs by keeping only the minimum ID for each unique normalized question.
    """
    subquery = (
        db.query(func.min(FAQModel.id))
        .group_by(func.lower(func.trim(FAQModel.question)))
        .subquery()
    )
    # Delete records whose ID is not in the subquery of minimum IDs
    db.query(FAQModel).filter(FAQModel.id.notin_(subquery)).delete(synchronize_session=False)
    db.commit()

def fetch_faqs(db: Session) -> List[FAQModel]:
    """
    Get all FAQs.
    """
    return db.query(FAQModel).all()

def all_messages(db: Session) -> List[MessageModel]:
    """
    Get all chat logs ordered by creation time descending.
    """
    return db.query(MessageModel).order_by(MessageModel.created_at.desc()).all()
