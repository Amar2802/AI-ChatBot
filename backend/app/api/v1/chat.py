import uuid
import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.chat import ChatRequest
from app.core.config import settings
from app.services import chat_service, nlp_service
from app.services.retriever_service import retriever_service
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/chat")
def chat(inp: ChatRequest, db: Session = Depends(get_db)):
    """
    Primary chat endpoint. Yields metadata, FAQ direct answers, or streams fallback LLM tokens in real-time.
    """
    sid = inp.session_id or str(uuid.uuid4())
    # Log user message
    chat_service.add_message(db, sid, "user", inp.message)
    
    # Build history context
    history = chat_service.get_recent_messages(db, sid, limit=settings.MAX_TURNS_MEMORY)

    
    # Retrieve FAQ candidates
    candidates = retriever_service.search(db, inp.message)
    
    # Choose response mode
    direct, score, mode = nlp_service.choose_response(inp.message, candidates)

    def event_generator():
        # Yield initial metadata
        yield json.dumps({
            "session_id": sid,
            "mode": mode,
            "similarity": score,
            "candidates": [{"id": i, "question": q, "score": s} for i, q, _, s in candidates],
            "chunk": "",
            "done": False
        }) + "\n"

        if direct:
            # Yield FAQ direct match
            mid = chat_service.add_message(db, sid, "assistant", direct)
            yield json.dumps({
                "message_id": mid,
                "chunk": direct,
                "done": True
            }) + "\n"
        else:
            # Stream LLM generation fallback
            full_text = []
            try:
                for token in nlp_service.generate_answer_stream(inp.message, history, candidates):
                    full_text.append(token)
                    yield json.dumps({
                        "chunk": token,
                        "done": False
                    }) + "\n"
            except Exception as e:
                print(f"Error in stream generation: {e}")
                yield json.dumps({
                    "chunk": "\n[Generation error, falling back...]",
                    "done": False
                }) + "\n"

            # Finalize and log assistant response
            final_answer = "".join(full_text)
            mid = chat_service.add_message(db, sid, "assistant", final_answer)
            yield json.dumps({
                "message_id": mid,
                "done": True
            }) + "\n"

    # Reference settings for turns limit within route
    # Bind settings property inside the service for convenience
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
