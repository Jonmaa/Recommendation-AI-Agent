# ============================================================
# vector_store.py - FAISS Vector Store for Product Embeddings
# ============================================================
# Builds and queries a FAISS index using Sentence-Transformer
# embeddings. Each product and each "co-purchase pattern" is
# stored as a vector, enabling fast similarity retrieval for
# the RAG pipeline.
# ============================================================

import json
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

from database import Product, get_all_products, get_co_purchased_products, PRODUCTS


# ----------------------------------------------------------------
# Embedding Model Configuration
# ----------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast & effective
EMBEDDING_DIM = 384


class ProductVectorStore:
    """
    FAISS-backed vector store that indexes:
      1. Individual product descriptions  (product index)
      2. Co-purchase pattern documents    (pattern index)

    This dual-index design supports the RAG retrieval step:
    given a purchased product, the agent retrieves the most
    similar co-purchase patterns and product descriptions to
    build context for the LLM.
    """

    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # --- Product Index ---
        self.product_index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.product_id_map: List[str] = []          # position → product_id
        self.product_texts: Dict[str, str] = {}      # product_id → text

        # --- Co-Purchase Pattern Index ---
        self.pattern_index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.pattern_id_map: List[str] = []           # position → source product_id
        self.pattern_docs: Dict[str, str] = {}        # source product_id → doc text

        self._build_indices()

    # ----------------------------------------------------------------
    # Index Construction
    # ----------------------------------------------------------------

    def _build_indices(self) -> None:
        """Generate embeddings and populate both FAISS indices."""
        products = get_all_products()

        # 1. Build the product index
        product_texts: List[str] = []
        for product in products:
            text = product.to_text()
            product_texts.append(text)
            self.product_id_map.append(product.id)
            self.product_texts[product.id] = text

        product_embeddings = self._encode(product_texts)
        self.product_index.add(product_embeddings)

        # 2. Build the co-purchase pattern index
        pattern_texts: List[str] = []
        for product in products:
            co_purchased = get_co_purchased_products(product.id)
            if not co_purchased:
                continue

            # Create a rich text document describing the co-purchase pattern
            doc = self._build_pattern_document(product, co_purchased)
            pattern_texts.append(doc)
            self.pattern_id_map.append(product.id)
            self.pattern_docs[product.id] = doc

        if pattern_texts:
            pattern_embeddings = self._encode(pattern_texts)
            self.pattern_index.add(pattern_embeddings)

        print(f"[VectorStore] Indexed {self.product_index.ntotal} products "
              f"and {self.pattern_index.ntotal} co-purchase patterns in FAISS.")

    def _build_pattern_document(
        self, source_product: Product, co_purchased: Dict[str, int]
    ) -> str:
        """
        Create a natural-language document that describes what other
        users bought along with `source_product`. This document is
        embedded and stored in the pattern index for RAG retrieval.
        """
        lines = [
            f"Users who bought '{source_product.name}' ({source_product.category}) "
            f"also frequently purchased the following products:"
        ]
        for pid, count in co_purchased.items():
            p = PRODUCTS.get(pid)
            if p:
                lines.append(
                    f"  - {p.name} (Category: {p.category}, "
                    f"Price: ${p.price:.2f}) — bought together {count} time(s). "
                    f"{p.description}"
                )
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Encoding Helpers
    # ----------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into normalized float32 embeddings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    # ----------------------------------------------------------------
    # Retrieval Methods
    # ----------------------------------------------------------------

    def search_similar_products(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Search the product index for the most similar products.
        Returns list of (product_id, score, product_text).
        """
        query_vec = self._encode([query])
        scores, indices = self.product_index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            pid = self.product_id_map[idx]
            results.append((pid, float(score), self.product_texts[pid]))
        return results

    def search_co_purchase_patterns(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[str, float, str]]:
        """
        Search the co-purchase pattern index for the most relevant
        purchase patterns.
        Returns list of (source_product_id, score, pattern_document).
        """
        if self.pattern_index.ntotal == 0:
            return []

        query_vec = self._encode([query])
        scores, indices = self.pattern_index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            pid = self.pattern_id_map[idx]
            results.append((pid, float(score), self.pattern_docs[pid]))
        return results

    def retrieve_context_for_product(
        self, product_id: str, top_k_patterns: int = 3, top_k_products: int = 5
    ) -> str:
        """
        RAG retrieval step: given a purchased product, retrieve
        relevant co-purchase patterns and similar products to form
        the context that will be passed to the LLM.
        """
        product = PRODUCTS.get(product_id)
        if not product:
            return "Product not found."

        query_text = product.to_text()

        # Retrieve co-purchase patterns
        patterns = self.search_co_purchase_patterns(query_text, top_k=top_k_patterns)

        # Retrieve similar products
        similar = self.search_similar_products(query_text, top_k=top_k_products + 1)
        # Exclude the product itself from similar results
        similar = [(pid, s, t) for pid, s, t in similar if pid != product_id][:top_k_products]

        # Build the RAG context document
        context_parts = [
            "=== PURCHASED PRODUCT ===",
            f"Product: {product.name}",
            f"Category: {product.category}",
            f"Description: {product.description}",
            f"Price: ${product.price:.2f}",
            f"Tags: {', '.join(product.tags)}",
            "",
        ]

        if patterns:
            context_parts.append("=== CO-PURCHASE PATTERNS (from user history) ===")
            for pid, score, doc in patterns:
                context_parts.append(f"[Relevance: {score:.3f}]")
                context_parts.append(doc)
                context_parts.append("")

        if similar:
            context_parts.append("=== SIMILAR PRODUCTS (by embedding similarity) ===")
            for pid, score, text in similar:
                p = PRODUCTS[pid]
                context_parts.append(
                    f"[Similarity: {score:.3f}] {p.name} — ${p.price:.2f} — {p.description}"
                )

        return "\n".join(context_parts)
