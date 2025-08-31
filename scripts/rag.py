# scripts/rag.py
from __future__ import annotations
from typing import List, Optional
import streamlit as st
from pathlib import Path
import os

# Vector DB & embeddings
from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# LLM via Ollama
from ollama import Client

# ---------- Config ----------
BASE_DIR     = Path(__file__).resolve().parent.parent   
PERSIST_DIR  = str(BASE_DIR / "chroma_store")           
COLLECTION   = "gouvernance"
EMB_MODEL    = "intfloat/multilingual-e5-base"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
ollama_client = Client(host=OLLAMA_HOST)
CTX_TOKENS   = 1024


# ---------- Embeddings ----------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------- Vector store ----------
vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

print("🔗 Chroma path:", PERSIST_DIR)
print("📦 Docs in collection:", vectorstore._collection.count())

# ---------- Retrieval ----------
def search_docs(query: str, k: int = 4, project_id: Optional[int] = None,
                use_mmr: bool = True, fetch_k: Optional[int] = None) -> List[Document]:
    """
    Retrieve top-k docs from Chroma using native API (bypassing LangChain wrapper).
    Filter by project_id if given.
    """
    filt = {"project_id": str(project_id)} if project_id not in (None, "", "None") else None
    print(f"🔎 Debug: query='{query}' | where={filt}")

    # Embed query
    q_emb = embeddings.embed_query(query)

    # Query Chroma directly
    res = vectorstore._collection.query(
        query_embeddings=[q_emb],
        n_results=int(k),
        where=filt if filt else None,
        include=["documents", "metadatas"]
    )

    docs = []
    if res and res.get("documents"):
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            docs.append(Document(page_content=doc, metadata=meta))

    return docs

# ---------- Generation ----------
def generate_answer(
    query: str,
    docs: List[Document],
    *,
    num_predict: int = 256,
    max_context_chars: int = 6000,
    temperature: float = 0.2
) -> str:
    """
    Call Ollama to generate a concise French answer.
    - Truncates context by characters for speed.
    - Limits num_predict to reduce latency.
    """
    raw_context = "\n\n".join(
        [getattr(d, "page_content", str(d)) for d in docs]
    ) if docs else ""

    context = raw_context[:max_context_chars] if raw_context else "(Aucun contexte pertinent trouvé.)"

    prompt = f"""Vous êtes un assistant. Répondez en français de manière très concise (3–6 points max).
N'inventez pas. Si le contexte est insuffisant, dites-le explicitement.

Question :
{query}

Contexte :
{context}

Réponse (courte) :
"""

    resp = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": CTX_TOKENS,
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
        stream=False,
    )
    return (resp.get("message", {}) or {}).get("content", "") or ""

# ---------- Public API ----------
def answer_query(
    question: str,
    project_id: Optional[int] = None,
    *,
    k: int = 4,
    use_mmr: bool = True,
    max_context_chars: int = 6000,
    num_predict: int = 256,
    temperature: float = 0.2
) -> str:
    """
    End-to-end RAG:
    - retrieval (MMR optional, filtered by project_id)
    - generation with limited context + limited output tokens
    Always returns a string (possibly empty), never None.
    """
    try:
        docs = search_docs(
            query=question,
            k=int(k),
            project_id=project_id,
            use_mmr=bool(use_mmr),
        )
    except Exception as e:
        return f"[RAG retrieval error] {e}"

    try:
        ans = generate_answer(
            query=question,
            docs=docs,
            num_predict=int(num_predict),
            max_context_chars=int(max_context_chars),
            temperature=float(temperature),
        )
    except Exception as e:
        return f"[LLM generation error] {e}"

    return ans or ""

def generate_answer_stream(
    query: str,
    docs: List[Document],
    *,
    num_predict: int = 256,
    max_context_chars: int = 6000,
    temperature: float = 0.2
) -> str:
    """Stream LLM answer progressively into Streamlit."""
    raw_context = "\n\n".join(
        [getattr(d, "page_content", str(d)) for d in docs]
    ) if docs else ""
    context = raw_context[:max_context_chars] if raw_context else "(No relevant context found.)"

    prompt = f"""You are an assistant. Answer in French, concise and clear.
Do NOT invent facts. If the context is insufficient, explicitly say so.

Question:
{query}

Context:
{context}

Answer:
"""

    # Create an empty container where the text will appear progressively
    container = st.empty()
    full_answer = ""

    # Stream response from Ollama
    for chunk in ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": CTX_TOKENS,
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
        stream=True,
    ):
        content = chunk["message"]["content"]
        full_answer += content
        container.markdown(full_answer)   # progressively update UI

    return full_answer

if __name__ == "__main__":
    print("Vérifiez le nombre de documents dans la collection...")
    try:
        stats = vectorstore._collection.count()
        print("Nombre de documents dans Chroma:", stats)
    except Exception as e:
        print("Erreur lors de l'interrogation de la collection:", e)
