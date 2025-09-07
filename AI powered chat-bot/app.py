from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from typing import Optional
from db import DB
from retriever import FAQRetriever
from nlp import generate_answer, choose_response
from config import PUBLIC_DIR, MAX_TURNS_MEMORY
app = FastAPI(title="AI Chatbot")
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True),name="static")
db = DB()
retriever = FAQRetriever(db)
class ChatIn(BaseModel):
	session_id: Optional[str] = None
	message: str

class FeedbackIn(BaseModel):
	session_id: str
	message_id: int
	rating: int # 1 or -1
	note: Optional[str] = None

@app.get("/health")
def health():
	return {"ok": True}

@app.post("/chat")
def chat(inp: ChatIn):
	sid = inp.session_id or str(uuid.uuid4())
	db.ensure_session(sid)
	# Log user message
	db.add_message(sid, "user", inp.message)
	# Build short history for context
	history = db.get_recent_messages(sid, limit=MAX_TURNS_MEMORY)
	# Retrieve FAQ candidates
	candidates = retriever.search(inp.message)
	# Decide direct vs. fallback
	direct, score, mode = choose_response(inp.message, candidates)
	if direct:
		answer = direct
	else:
		answer = generate_answer(inp.message, history, candidates)
	mid = db.add_message(sid, "assistant", answer)
	return {
		"session_id": sid,
		"message_id": mid,
		"mode": mode,
		"similarity": score,
		"answer": answer,
		"candidates": [
			{"id": i, "question": q, "score": s} for i, q, _, s in candidates
		]
	}

@app.post("/feedback")
def feedback(inp: FeedbackIn):
	db.log_feedback(inp.session_id, inp.message_id, inp.rating, inp.note)
	return {"ok": True}

class BulkFAQIn(BaseModel):
	faqs: list

@app.post("/faqs/bulk")
def faqs_bulk(inp: BulkFAQIn):
	pairs = []
	for item in inp.faqs:
		q = item.get("question")
		a = item.get("answer")
		if not q or not a:
			raise HTTPException(400, "Each FAQ needs 'question' and 'answer'")
		pairs.append((q, a))
	db.insert_faqs(pairs)
	retriever.reload()
	return {"inserted": len(pairs)}

@app.get("/logs")
def logs():
	rows = db.all_messages()
	return {"count": len(rows), "messages": rows}
