from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import uuid
from typing import Optional
from db import DB
from retriever import FAQRetriever
from nlp import generate_answer, choose_response
from config import PUBLIC_DIR, MAX_TURNS_MEMORY, FAQ_PATH
app = FastAPI(title="AI Chatbot")
db = DB()


def _bootstrap_faqs():
    db.dedupe_faqs()
    if db.faq_count() == 0 and FAQ_PATH.exists():
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        pairs = [(i["question"], i["answer"]) for i in items]
        db.replace_faqs(pairs)


_bootstrap_faqs()
retriever = FAQRetriever(db)

from config import USE_LOCAL_LLM, OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEN_MODEL
if OPENAI_API_KEY and not USE_LOCAL_LLM:
    print("\n" + "="*60)
    print("🤖 CHATBOT STARTING IN OPENAI API MODE (Fast & Accurate)")
    print(f"Model: {OPENAI_MODEL}")
    print("="*60 + "\n")
elif GEMINI_API_KEY and not USE_LOCAL_LLM:
    print("\n" + "="*60)
    print("🤖 CHATBOT STARTING IN GEMINI API MODE (Fast & Accurate)")
    print("Model: gemini-2.5-flash")
    print("="*60 + "\n")
else:
    print("\n" + "="*60)
    print("⚠️ CHATBOT STARTING IN LOCAL CPU MODE (Slow / Fallback)")
    print(f"Model: {GEN_MODEL}")
    if not OPENAI_API_KEY and not GEMINI_API_KEY:
        print("Tip: Add OPENAI_API_KEY or GEMINI_API_KEY in your .env file to enable high-speed API generation!")
    print("="*60 + "\n")


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

app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
