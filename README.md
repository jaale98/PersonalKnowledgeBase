# Personal Knowledge Base

A small FastAPI + Postgres app for keeping notes with tags and searching them using natural language.

---

## What It Does

- Create, edit, and delete notes
- Tag notes (supports multiple tags per note)
- Search notes using natural language (via OpenAI embeddings)
- Optional hybrid search that mixes text matching and vector similarity
- Works entirely locally with Docker or a Python virtual environment

---

## Getting Started

### 1. Clone and set up
```bash
git clone https://github.com/jaale98/PersonalKnowledgeBase.git
cd PersonalKnowledgeBase
cp .env.example .env
```

Then open `.env` and fill in:
```
OPENAI_API_KEY=sk-xxxx
DATABASE_URL=postgresql+psycopg://pkb:pkb@db:5432/pkb
```

---

### 2. Start it with Docker
```bash
docker compose up -d
```

This runs:
- `db` — PostgreSQL with pgvector
- `pgweb` — database browser at http://localhost:8081
- `app` — FastAPI server at http://localhost:8000

---

### 3. Load the database schema
If it’s the first run:
```bash
docker exec -it pkb-db psql -U pkb -d pkb -f /schema.sql
```

---

### 4. Visit the app
Go to [http://localhost:8000](http://localhost:8000)

From there you can:
- Add notes with tags
- Search by meaning instead of exact keywords
- Filter by tags
- Edit or delete what you’ve added

---

## Run locally (without Docker)
If you’d rather just use Python directly:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Notes

- The database uses the `pgvector` extension for semantic search.
- Hybrid mode adds PostgreSQL’s built-in full-text search for better matching.
- There’s no authentication — it’s just for local use.
- You can view or edit everything in `static/index.html`.

---

## Example Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"q":"notes about databases", "limit":5}'
```

---

That’s it. A simple, self-contained notes app with tags and search.