import json
from db import DB
from config import FAQ_PATH

db = DB()
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)
pairs = [(i["question"], i["answer"]) for i in items]
db.replace_faqs(pairs)
print(f"Loaded {len(pairs)} FAQs (replaced existing entries).")
