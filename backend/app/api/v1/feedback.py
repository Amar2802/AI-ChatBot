from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.feedback import FeedbackRequest
from app.services import chat_service

router = APIRouter()

@router.post("/feedback")
def feedback(inp: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Log user feedback rating (1 or -1) and optional note for a specific message.
    """
    chat_service.log_feedback(db, inp.session_id, inp.message_id, inp.rating, inp.note)
    return {"ok": True}
