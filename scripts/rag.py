from __future__ import annotations
from typing import List, Optional
import streamlit as st
from pathlib import Path

# Vector DB & embeddings

from langchain_core.documents import Document
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI

# LLM via OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.2
)

# Config 
BASE_DIR     = Path(__file__).resolve().parent.parent   
PERSIST_DIR  = str(BASE_DIR / "chroma_store")           
COLLECTION   = "demo_docs"

# Vector store 
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION)

#  Retrieval
def search_docs(query: str, k: int = 4, project_id: Optional[int] = None) -> List[Document]:
    where = {"project_id": project_id} if project_id is not None else None

    res = collection.query(
        query_texts=[query],
        n_results=k,
        where=where
    )

    docs = []
    for d, m in zip(res["documents"][0], res["metadatas"][0]):
        docs.append(Document(page_content=d, metadata=m))
    return docs

# Generation
def generate_answer(query:str, docs: List[Document]) -> str:
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Réponds en français, de manière concise.
N'invente rien.

Question :
{query}

Contexte :
{context}

Réponse :
"""
    return llm.invoke(prompt).content

# Public API
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
    )
    except Exception as e:
        return f"[RAG retrieval error] {e}"

    try:
        ans = generate_answer(
        query=question,
        docs=docs
    )

    except Exception as e:
        return f"[LLM generation error] {e}"

    return ans or ""


if __name__ == "__main__":
    print("Vérifiez le nombre de documents dans la collection...")
    try:
        stats = collection.count()
        print("Nombre de documents dans Chroma:", stats)
    except Exception as e:
        print("Erreur lors de l'interrogation de la collection:", e)
