import json
from db import DB
from config import FAQ_PATH
db = DB()
with open(FAQ_PATH, 'r', encoding='utf-8') as f:
	items = json.load(f)
	db.insert_faqs([(i['question'], i['answer']) for i in items])
	print(f"Inserted {len(items)} FAQs.")
