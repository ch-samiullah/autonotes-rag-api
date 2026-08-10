from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import uuid
from datetime import datetime

app = FastAPI(
    title="AutoNotes Knowledge Base API",
    description="RAG-powered document QA system.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
groq_client = Groq(api_key=GROQ_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    min_relevance_score: float = 0.3

class QueryResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    model_used: str
    confidence: str
    timestamp: str

class AddTextRequest(BaseModel):
    text: str
    source_name: str = "manual_input"

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedding_model: str
    status: str

def build_rag_prompt(question: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join([
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in context_chunks
    ])
    return f"""You are a helpful assistant for AutoNotes Knowledge Base.

STRICT RULES:
1. Answer ONLY from the provided context below
2. If answer is not in context, say "Mujhe is topic pe relevant information nahi mili"
3. Always mention which source you used
4. Be concise but complete

CONTEXT:
{context}

USER QUESTION: {question}

Answer (mention source):"""

def calculate_confidence(distances: list) -> str:
    if not distances:
        return "low"
    avg_similarity = 1 - (sum(distances) / len(distances))
    if avg_similarity > 0.7:
        return "high"
    elif avg_similarity > 0.4:
        return "medium"
    else:
        return "low"

@app.get("/")
def home():
    return {
        "name": "AutoNotes Knowledge Base API",
        "version": "1.0.0",
        "status": "running",
        "total_chunks_indexed": collection.count(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/documents/add-text")
def add_text_document(request: AddTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text empty nahi ho sakta")
    chunks = splitter.split_text(request.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text se chunks nahi bane")
    chunk_ids = [f"{request.source_name}_{uuid.uuid4().hex[:8]}" for _ in chunks]
    metadatas = [
        {
            "source": request.source_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "added_at": datetime.now().isoformat()
        }
        for i in range(len(chunks))
    ]
    collection.add(documents=chunks, ids=chunk_ids, metadatas=metadatas)
    return {
        "status": "success",
        "source": request.source_name,
        "chunks_created": len(chunks),
        "message": f"{len(chunks)} chunks indexed successfully"
    }

@app.post("/rag/query", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question empty nahi ho sakta")
    total_docs = collection.count()
    if total_docs == 0:
        raise HTTPException(status_code=404, detail="Knowledge base empty hai")
    results = collection.query(
        query_texts=[request.question],
        n_results=min(request.top_k, total_docs),
        include=["documents", "metadatas", "distances"]
    )
    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]
    retrieved_distances = results["distances"][0]
    filtered_chunks = []
    filtered_distances = []
    for doc, meta, dist in zip(retrieved_docs, retrieved_meta, retrieved_distances):
        similarity = 1 - dist
        if similarity >= request.min_relevance_score:
            filtered_chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "similarity": round(similarity, 3)
            })
            filtered_distances.append(dist)
    if not filtered_chunks:
        return QueryResponse(
            answer="Relevant information nahi mili.",
            sources=[],
            chunks_used=0,
            model_used="none",
            confidence="low",
            timestamp=datetime.now().isoformat()
        )
    prompt = build_rag_prompt(request.question, filtered_chunks)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Only use provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        model_used = response.model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    return QueryResponse(
        answer=answer,
        sources=[{"source": c["source"], "similarity_score": c["similarity"], "preview": c["text"][:100] + "..."} for c in filtered_chunks],
        chunks_used=len(filtered_chunks),
        model_used=model_used,
        confidence=calculate_confidence(filtered_distances),
        timestamp=datetime.now().isoformat()
    )

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    return StatsResponse(
        total_chunks=collection.count(),
        collection_name="knowledge_base",
        embedding_model="all-MiniLM-L6-v2",
        status="healthy"
    )

@app.delete("/documents/clear")
def clear_knowledge_base():
    global collection
    chroma_client.delete_collection("knowledge_base")
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return {"status": "cleared", "message": "Knowledge base reset ho gaya"}