from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from db import DB
from config import EMBEDDING_MODEL, TOP_K

class FAQRetriever:
    def __init__(self, db: DB):
        self.db = db
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._index()
        rows = self.db.fetch_faqs()
        self.ids = [r[0] for r in rows]
        self.questions = [r[1] for r in rows]
        self.answers = [r[2] for r in rows]
        if self.questions:
            self.embeddings = self.model.encode(self.questions,convert_to_numpy=True, normalize_embeddings=True)
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            def reload(self):
                self._index()
                def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[int, str,
                                                                               str, float]]:
                        if len(self.questions) == 0:
                             return []
                        q_emb = self.model.encode([query], convert_to_numpy=True,normalize_embeddings=True)
                        sims = cosine_similarity(q_emb, self.embeddings)[0]
                        idxs = np.argsort(-sims)[:top_k]
                        return [(self.ids[i], self.questions[i], self.answers[i],float(sims[i])) for i in idxs]