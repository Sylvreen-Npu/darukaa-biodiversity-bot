"""
Retrieval layer.

For the hackathon submission this uses a TF-IDF vector search (scikit-learn)
over a small, curated set of knowledge chunks (knowledge/sources.json), plus
metadata filtering by topic/region. This keeps the system fully offline and
reviewable, while remaining a genuine embedding + vector-similarity retrieval
pipeline (swap in `text-embedding-3-large` / Chroma per the design doc if you
want semantic embeddings instead of TF-IDF — the interface below is unchanged
either way).
"""
import json
import os
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "sources.json")


class Retriever:
    def __init__(self, knowledge_path: str = KNOWLEDGE_PATH):
        with open(knowledge_path, "r") as f:
            self.chunks: List[Dict] = json.load(f)
        self._texts = [c["text"] for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(
        self,
        query: str,
        topic_filter: Optional[List[str]] = None,
        region_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        candidates = []
        for idx, chunk in enumerate(self.chunks):
            score = sims[idx]
            if topic_filter and chunk["topic"] not in topic_filter:
                score *= 0.3  # soft penalty rather than hard exclusion
            if region_filter and chunk.get("region") and chunk["region"] != region_filter:
                score *= 0.7
            candidates.append((score, chunk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in candidates[:top_k] if score > 0]
