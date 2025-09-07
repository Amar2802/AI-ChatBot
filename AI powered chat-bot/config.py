from pathlib import Path
# Model IDs (change if you like)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base" # small, CPU-friendly
# Retrieval settings
TOP_K = 3
SIM_THRESHOLD = 0.55 # tune for your data
# Conversation settings
MAX_TURNS_MEMORY = 6 # last N user-bot messages kept as context
# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "chatbot.db"
FAQ_PATH = DATA_DIR / "faqs.json"
PUBLIC_DIR = BASE_DIR / "public"
