# AI Product Recommendation Agent

An intelligent product recommendation system that uses **RAG (Retrieval-Augmented Generation)** with **FAISS** vector search and **Groq** (Llama 3.3 70B) to suggest products based on collaborative purchase patterns.

## How It Works

```
User buys a product
        │
        ▼
┌─────────────────────────────┐
│  FAISS Vector Store (RAG)   │
│  ┌────────────────────────┐ │
│  │ Co-purchase patterns   │ │  ← "Users who bought X also bought Y, Z"
│  │ Product embeddings     │ │  ← Semantic similarity between products
│  └────────────────────────┘ │
└─────────────┬───────────────┘
              │ Retrieved context
              ▼
┌─────────────────────────────┐
│  Groq LLM (Generation)      │
│  System prompt + context    │
│  → Natural language recs    │
└─────────────┬───────────────┘
              │
              ▼
   "Other users also bought..."
```

1. **Purchase recorded** directly in the database — new users are created automatically and purchases persist across restarts
2. **FAISS retrieval** finds co-purchase patterns and similar products using vector similarity
3. **Context injection** — retrieved data is passed to the LLM as context
4. **Groq LLM generates** a natural-language recommendation with explanations

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Interactive CLI application |
| `database.py` | Product catalog (30 products), user purchase history, and self-updating persistence |
| `vector_store.py` | FAISS index builder & retriever using Sentence-Transformers |
| `recommendation_agent.py` | RAG orchestrator + Groq LLM integration |
| `.env` | API key configuration |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key (free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=gsk_your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Run

```bash
python main.py
```

## Commands

| Command | Description |
|---------|-------------|
| `catalog` | Display all 30 products |
| `buy P001` | Buy a product and receive AI recommendations |
| `purchases` | Show your purchase history with totals |
| `search yoga` | Semantic search across the product catalog |
| `chat <message>` | Ask the agent follow-up questions |
| `reset` | Clear conversation history |
| `help` | Show available commands |
| `quit` | Exit |

## Product Categories

- **Footwear** — Nike, Adidas, New Balance, Puma (5 products)
- **Fitness** — Dumbbells, kettlebells, barbells, resistance bands, yoga mat (5 products)
- **Sports** — Football, basketball, tennis, volleyball, boxing gloves (5 products)
- **Food & Nutrition** — Protein, granola, energy bars, olive oil, quinoa, matcha, creatine (7 products)
- **Technology** — AirPods, Garmin watch, Fitbit, Sony headphones, GoPro, Kindle, mouse, tablet (8 products)

## Example

```
>>> buy P001

Purchase confirmed! You bought: Nike Air Max 90 ($129.99)

┌─ AI Recommendation ─────────────────────────────┐
│                                                  │
│  Other users also bought...                      │
│                                                  │
│  1. Whey Protein Isolate ($54.99)                │
│     Popular among fitness enthusiasts who also   │
│     invest in quality running shoes.             │
│                                                  │
│  2. Garmin Forerunner 265 ($449.99)              │
│     Runners who buy Nike Air Max often pair them │
│     with a GPS watch to track their performance. │
│                                                  │
│  3. Energy Bar Variety Pack ($24.99)             │
│     A quick fuel option favoured by active       │
│     customers on the go.                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.10+**
- **FAISS** (Facebook AI Similarity Search) — vector similarity search
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — product embedding generation
- **Groq** (`llama-3.3-70b-versatile`) — fast, free LLM inference via OpenAI-compatible API
- **Rich** — terminal UI formatting

## Persistence

User purchases are stored directly in `database.py`. When a new user buys a product, the `USER_PURCHASES` list in the source file is automatically updated. This means:

- **Returning users** are recognized by name and shown their purchase history
- **New users** are created with an auto-generated ID (U016, U017, ...)
- **All purchases persist** across application restarts — no external database needed
