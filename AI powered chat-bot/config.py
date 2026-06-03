import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Model IDs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Default to a lighter 0.5B model for local CPU fallback (much faster than 1.5B)
GEN_MODEL = os.getenv("LOCAL_GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# OpenAI API config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Gemini API config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None

use_local_llm_str = os.getenv("USE_LOCAL_LLM", "").strip().lower()
if use_local_llm_str in ("true", "1", "yes"):
    USE_LOCAL_LLM = True
elif use_local_llm_str in ("false", "0", "no"):
    USE_LOCAL_LLM = False
else:
    USE_LOCAL_LLM = (OPENAI_API_KEY is None) and (GEMINI_API_KEY is None)


# Retrieval settings
TOP_K = 3
SIM_THRESHOLD = 0.65      # Direct FAQ answer when similarity is this high
FAQ_CONTEXT_MIN = 0.40    # Include FAQ in LLM context above this score

# Conversation settings
MAX_TURNS_MEMORY = 6      # Last N user-bot messages kept as context

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
DB_PATH = BASE_DIR / "chatbot.db"
FAQ_PATH = DATA_DIR / "faqs.json"
PUBLIC_DIR = BASE_DIR / "Public"

