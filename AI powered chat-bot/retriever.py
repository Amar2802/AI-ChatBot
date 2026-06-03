from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from db import DB
from config import EMBEDDING_MODEL, TOP_K
import re

class FAQRetriever:
    def __init__(self, db: DB):
        self.db = db
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._index() # Load and index FAQs
    
    def _index(self):
        rows = self.db.fetch_faqs()
        self.ids = [r[0] for r in rows]
        self.questions = [r[1] for r in rows]
        self.answers = [r[2] for r in rows]
        if self.questions:
            # Embed question + answer so paraphrases and keyword-style queries match better
            corpus = [
                f"{q} {a}" for q, a in zip(self.questions, self.answers)
            ]
            self.embeddings = self.model.encode(
                corpus, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
    
    def reload(self):
        self._index()

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[int, str, str, float]]:
        if len(self.questions) == 0:
             self._index()
        if len(self.questions) == 0:
             return []
        
        # Clean user query for basic keyword-based checks
        q_clean = re.sub(r'[^\w\s]', '', query.lower())
        q_words = set(q_clean.split())
        stop_words = {"what", "are", "your", "how", "can", "i", "do", "you", "offer", "is", "a", "an", "the", "on", "in", "to", "for", "with", "my", "of", "and"}
        q_keywords = q_words - stop_words
        
        # Dense vector similarity search
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        
        # Apply hybrid search: boost score if there is key term overlap
        boosted_sims = sims.copy()
        if q_keywords:
            for i, question in enumerate(self.questions):
                q_idx_clean = re.sub(r'[^\w\s]', '', question.lower())
                q_idx_words = set(q_idx_clean.split())
                overlapping_keywords = q_keywords.intersection(q_idx_words)
                if overlapping_keywords:
                    # Provide a boost proportional to matching keywords (max boost of 0.2)
                    boost = min(0.2, len(overlapping_keywords) * 0.08)
                    # Cap boosted similarity to 1.0
                    boosted_sims[i] = min(1.0, boosted_sims[i] + boost)
        
        # Sort indices by boosted similarity; skip duplicate questions
        idxs = np.argsort(-boosted_sims)
        seen_questions = set()
        results = []
        for i in idxs:
            q_norm = self.questions[i].strip().lower()
            if q_norm in seen_questions:
                continue
            seen_questions.add(q_norm)
            results.append(
                (self.ids[i], self.questions[i], self.answers[i], float(boosted_sims[i]))
            )
            if len(results) >= top_k:
                break
        return results