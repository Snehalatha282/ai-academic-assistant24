import os
import re
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AcademicRAGEngine:
    """
    RAG & Semantic Retrieval Engine for Academic Documents.
    Provides robust semantic retrieval with exact page citations.
    """

    def __init__(self, use_gemini_embeddings: bool = False, api_key: Optional[str] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.use_gemini_embeddings = use_gemini_embeddings
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Builds retrieval indices from text chunks."""
        self.chunks = chunks
        if not chunks:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.dense_embeddings = None
            return

        texts = [c["text"] for c in chunks]

        # 1. Build TF-IDF Semantic Index (fast, zero-failure, offline-ready)
        # Uses n-grams (1, 2) to capture academic terms like "bubble sort", "virtual memory", "page table"
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_df=0.95,
            min_df=1
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # 2. Try Gemini Dense Embeddings if configured
        if self.use_gemini_embeddings and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                dense_vectors = []
                # Batch in groups of 20
                for i in range(0, len(texts), 20):
                    batch = texts[i:i+20]
                    res = genai.embed_content(
                        model="models/text-embedding-004",
                        content=batch,
                        task_type="retrieval_document"
                    )
                    dense_vectors.extend(res['embedding'])
                self.dense_embeddings = np.array(dense_vectors)
            except Exception as e:
                print(f"Gemini embeddings initialization skipped: {e}")
                self.dense_embeddings = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most semantically relevant chunks with similarity scores.
        """
        if not self.chunks or self.tfidf_vectorizer is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        # Sparse TF-IDF Cosine Similarity
        query_vec = self.tfidf_vectorizer.transform([clean_query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Hybrid Dense Scoring if available
        if self.dense_embeddings is not None and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                q_res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=clean_query,
                    task_type="retrieval_query"
                )
                q_vec = np.array(q_res['embedding']).reshape(1, -1)
                dense_scores = cosine_similarity(q_vec, self.dense_embeddings).flatten()
                # Combined hybrid score (70% dense + 30% lexical keyword)
                scores = 0.7 * dense_scores + 0.3 * tfidf_scores
            except Exception:
                scores = tfidf_scores
        else:
            scores = tfidf_scores

        # Rank indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "rank": rank + 1,
                "score": round(score, 4),
                "text": chunk["text"],
                "page": chunk.get("page", 1),
                "source_doc": chunk.get("source_doc", "Document"),
                "chunk_id": chunk.get("chunk_id", idx + 1),
                "section": chunk.get("section", f"Page {chunk.get('page', 1)}")
            })

        return results

    def get_document_summary_chunks(self, limit: int = 4) -> List[Dict[str, Any]]:
        """Returns representative starting chunks for overview generation."""
        return self.chunks[:limit]
