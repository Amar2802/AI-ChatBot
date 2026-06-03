from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.faq import BulkFAQRequest
from app.services import chat_service
from app.services.retriever_service import retriever_service

router = APIRouter()

@router.post("/faqs/bulk")
def faqs_bulk(inp: BulkFAQRequest, db: Session = Depends(get_db)):
    """
    Bulk insert FAQs and force vector search indexing reload.
    """
    pairs = []
    for item in inp.faqs:
        if not item.question or not item.answer:
            raise HTTPException(400, "Each FAQ needs 'question' and 'answer'")
        pairs.append((item.question, item.answer))
    
    chat_service.insert_faqs(db, pairs)
    retriever_service.reload(db)
    return {"inserted": len(pairs)}

@router.get("/logs")
def logs(db: Session = Depends(get_db)):
    """
    Return all chat message logs.
    """
    rows = chat_service.all_messages(db)
    # Convert SQLAlchemy model instances to dicts or schemas for response serialization
    serialized_messages = [
        {
            "id": r.id,
            "session_id": r.session_id,
            "role": r.role,
            "text": r.text,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in rows
    ]
    return {"count": len(serialized_messages), "messages": serialized_messages}
