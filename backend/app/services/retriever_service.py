import re
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services import chat_service

class FAQRetrieverService:
    def __init__(self):
        self._model = None
        self.ids: List[int] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 384), dtype=np.float32)

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the sentence transformer model.
        """
        if self._model is None:
            print(f"Loading embedding model '{settings.EMBEDDING_MODEL}'...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._model

    def index(self, db: Session):
        """
        Load FAQs from DB and compute text embeddings.
        """
        rows = chat_service.fetch_faqs(db)
        self.ids = [r.id for r in rows]
        self.questions = [r.question for r in rows]
        self.answers = [r.answer for r in rows]

        if self.questions:
            # Embed question + answer so paraphrases and keyword-style queries match better
            corpus = [f"{q} {a}" for q, a in zip(self.questions, self.answers)]
            self.embeddings = self.model.encode(
                corpus, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)

    def reload(self, db: Session):
        """
        Force re-index FAQs.
        """
        self.index(db)

    def search(self, db: Session, query: str, top_k: int = None) -> List[Tuple[int, str, str, float]]:
        """
        Search FAQs using a hybrid dense vector similarity + keyword term boosting logic.
        """
        k = top_k or settings.TOP_K
        if not self.questions:
            self.index(db)
        if not self.questions:
            return []

        # Clean user query for basic keyword-based checks
        q_clean = re.sub(r'[^\w\s]', '', query.lower())
        q_words = set(q_clean.split())
        stop_words = {
            "what", "are", "your", "how", "can", "i", "do", "you", "offer", "is", 
            "a", "an", "the", "on", "in", "to", "for", "with", "my", "of", "and"
        }
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
            if len(results) >= k:
                break
        return results

# Singleton instance of retriever service
retriever_service = FAQRetrieverService()
