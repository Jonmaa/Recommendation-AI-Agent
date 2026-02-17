# ============================================================
# recommendation_agent.py - AI Recommendation Agent (RAG)
# ============================================================
# Orchestrates the full RAG pipeline:
#   1. User "buys" a product
#   2. FAISS retrieves co-purchase patterns & similar products
#   3. Retrieved context is injected into a prompt
#   4. Groq LLM generates a natural-language recommendation
# ============================================================

import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

from database import (
    Product,
    get_product,
    get_co_purchased_products,
    add_purchase,
    PRODUCTS,
)
from vector_store import ProductVectorStore

load_dotenv()


# ----------------------------------------------------------------
# System Prompt
# ----------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly and knowledgeable shopping assistant AI.
Your job is to recommend products to users based on what other customers
who bought the same item have also purchased.

You will receive:
1. The product the user just bought.
2. Co-purchase patterns retrieved from a vector database (FAISS) showing
   what other users who bought the same or similar products also purchased.
3. A list of similar products by embedding similarity.

RULES:
- Recommend 3 to 5 products maximum.
- Present recommendations under the heading "Other users also bought..."
- For each recommendation, briefly explain WHY it is relevant
  (e.g., "popular among runners", "great complement for gym workouts").
- Include the product name and price.
- Be conversational but concise.
- Do NOT recommend the product the user just bought.
- If possible, recommend products from different categories to offer variety.
- Answer in the same language the user writes in. Default to English."""


class RecommendationAgent:
    """
    AI Agent that uses Retrieval-Augmented Generation (RAG) with FAISS
    to provide personalised product recommendations.
    """

    def __init__(self, vector_store: ProductVectorStore):
        self.vector_store = vector_store
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.conversation_history: List[Dict[str, str]] = []

    # ----------------------------------------------------------------
    # Core Recommendation Flow
    # ----------------------------------------------------------------

    def recommend_after_purchase(
        self, username: str, product_id: str
    ) -> str:
        """
        Main entry point: processes a purchase and returns
        AI-generated recommendations.

        Steps:
          1. Record the purchase
          2. Retrieve RAG context from FAISS
          3. Build the augmented prompt
          4. Call the LLM for a natural-language recommendation
        """
        # Step 1 — Record the purchase
        product = get_product(product_id)
        if not product:
            return f"Error: Product '{product_id}' not found in catalog."

        add_purchase(username, product_id)

        # Step 2 — RAG Retrieval via FAISS
        rag_context = self.vector_store.retrieve_context_for_product(
            product_id, top_k_patterns=3, top_k_products=5
        )

        # Step 3 — Build augmented prompt
        user_message = (
            f"I just bought: {product.name} (${product.price:.2f}).\n\n"
            f"Based on the following purchase data and product similarities, "
            f"please recommend other products I might like.\n\n"
            f"--- RETRIEVED CONTEXT ---\n{rag_context}\n--- END CONTEXT ---"
        )

        # Step 4 — Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history,
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )

        assistant_reply = response.choices[0].message.content

        # Update conversation history for multi-turn support
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

        return assistant_reply

    # ----------------------------------------------------------------
    # Free-form Chat
    # ----------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Handle general questions from the user (e.g., asking for more
        details about a recommended product).
        """
        # Try to find products mentioned in the message for context
        extra_context = self._find_relevant_context(user_message)

        full_message = user_message
        if extra_context:
            full_message += f"\n\n--- ADDITIONAL CONTEXT ---\n{extra_context}\n--- END ---"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history,
            {"role": "user", "content": full_message},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )

        assistant_reply = response.choices[0].message.content

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

        return assistant_reply

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _find_relevant_context(self, text: str) -> str:
        """Use FAISS semantic search to find products related to the user query."""
        results = self.vector_store.search_similar_products(text, top_k=3)
        if not results:
            return ""

        lines = []
        for pid, score, _ in results:
            p = PRODUCTS[pid]
            lines.append(
                f"- {p.name} (${p.price:.2f}, {p.category}): {p.description}"
            )
        return "\n".join(lines)

    def reset_conversation(self) -> None:
        """Clear conversation history for a fresh session."""
        self.conversation_history.clear()
