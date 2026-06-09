from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import SessionLocal
from app.db.init_db import init_db
from app.services.retriever_service import retriever_service
from app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    setup_logging()
    
    # Bootstrap database schemas and seed default FAQs
    db = SessionLocal()
    try:
        init_db(db)
        # Pre-build vector retriever indices
        retriever_service.index(db)
    finally:
        db.close()
        
    # Pre-load local LLM model if enabled to avoid first-request latency
    if settings.USE_LOCAL_LLM:
        from app.services.nlp_service import _get_local_model_and_tokenizer
        print("Pre-loading local LLM model...")
        _get_local_model_and_tokenizer()
        print("Local LLM model pre-loaded successfully.")
        
    yield
    # Shutdown tasks (none needed)

app = FastAPI(title="AI Chatbot", lifespan=lifespan)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(api_router)


# Display LLM execution mode banner (Plain text to avoid Windows UnicodeEncodeErrors)
if settings.OPENAI_API_KEY and not settings.USE_LOCAL_LLM:
    print("\n" + "="*60)
    print("CHATBOT STARTING IN OPENAI API MODE (Fast & Accurate)")
    print(f"Model: {settings.OPENAI_MODEL}")
    print("="*60 + "\n")
elif settings.GEMINI_API_KEY and not settings.USE_LOCAL_LLM:
    print("\n" + "="*60)
    print("CHATBOT STARTING IN GEMINI API MODE (Fast & Accurate)")
    print("Model: gemini-2.5-flash")
    print("="*60 + "\n")
else:
    print("\n" + "="*60)
    print("CHATBOT STARTING IN LOCAL CPU MODE (Slow / Fallback)")
    print(f"Model: {settings.LOCAL_GEN_MODEL}")
    if not settings.OPENAI_API_KEY and not settings.GEMINI_API_KEY:
        print("Tip: Add OPENAI_API_KEY in your .env file to enable high-speed API generation!")
    print("="*60 + "\n")

