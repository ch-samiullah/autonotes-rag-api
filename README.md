# AutoNotes Knowledge Base API
## RAG + FastAPI + Groq — Production Level Project

---

## Real Problem
Company ke documents (HR policy, technical docs, product info) mein manually search karna slow aur inefficient hai. Koi bhi employee ya customer baar baar same questions poochta hai.

## Solution
RAG pipeline jo:
- Documents ko semantic chunks mein store karta hai
- Natural language questions ka accurate jawab deta hai
- Hallucination rokta hai source attribution se
- Groq (free) se blazing fast responses deta hai

---

## Algorithm

```
INDEX TIME (ek baar):
  Documents → RecursiveCharacterTextSplitter (500 chars, 50 overlap)
           → SentenceTransformer embedding (all-MiniLM-L6-v2, 384 dim)
           → ChromaDB (cosine similarity index)

QUERY TIME (har request):
  Question → embed (same model)
           → ChromaDB cosine similarity search
           → top-k chunks retrieve
           → min_relevance_score filter (hallucination prevention)
           → RAG prompt assembly (context + strict rules)
           → Groq LLM (llama-3.1-8b-instant, temp=0.1)
           → Answer + Sources + Confidence
```

---

## Setup — Local

```bash
# 1. Clone / folder mein jao
cd rag_project

# 2. Virtual environment banao
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# 3. Dependencies install karo
pip install -r requirements.txt

# 4. Groq API key set karo
# .env file mein apni key daalo
# https://console.groq.com → free account → API Keys

# 5. Server run karo
uvicorn main:app --reload

# 6. Sample data add karo (naye terminal mein)
python seed_data.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/documents/add-text` | Text document add karo |
| POST | `/rag/query` | Question pucho |
| GET | `/stats` | KB statistics |
| DELETE | `/documents/clear` | KB clear karo |
| GET | `/docs` | Swagger UI |

---

## Test Karo

```bash
# Document add karo
curl -X POST http://localhost:8000/documents/add-text \
  -H "Content-Type: application/json" \
  -d '{"text": "AutoNotes Pro ka price Rs. 999 per month hai.", "source_name": "pricing"}'

# Question pucho
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "AutoNotes ka price kya hai?", "top_k": 3}'
```

---

## Production Deployment — Railway (Free)

```bash
# 1. GitHub pe push karo
git init
git add .
git commit -m "RAG API initial commit"
git remote add origin YOUR_GITHUB_URL
git push origin main

# 2. railway.app pe jao → New Project → Deploy from GitHub
# 3. Environment Variables mein GROQ_API_KEY add karo
# 4. Deploy — automatic!
```

## Production Deployment — Docker

```bash
# Build
docker build -t autonotes-rag-api .

# Run
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  autonotes-rag-api
```

---

## Interview Points

- **Chunking overlap kyun:** Boundary pe context preserve karne ke liye
- **Cosine similarity:** Do vectors ka angle — 1 = same, 0 = unrelated
- **min_relevance_score:** Low similarity chunks filter karo → hallucination rokta hai
- **temperature=0.1:** Low temp = deterministic, less creative = less hallucination
- **Source attribution:** Har answer mein source → trust + debuggability
- **Persistent ChromaDB:** Server restart ke baad data safe
- **RAG vs Fine-tuning:** RAG updatable + cheap, fine-tuning better for style
