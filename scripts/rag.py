from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import os

# LangChain & Chroma
from langchain_core.documents import Document
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI

# -----------------------------
# Vérification clé OpenAI
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

# -----------------------------
# LLM (LangChain)
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)

# -----------------------------
# Paths & Chroma config
# -----------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(BASE_DIR / "chroma_store")
COLLECTION  = "demo_docs"

# -----------------------------
# Vector store
# -----------------------------
chroma_client = PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_collection(name=COLLECTION)

# -----------------------------
# Retrieval
# -----------------------------
def search_docs(
    query: str,
    k: int = 4,
    project_id: Optional[int] = None
) -> List[Document]:

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

# -----------------------------
# Generation
# -----------------------------
def generate_answer(query: str, docs: List[Document]) -> str:
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

# -----------------------------
# Public API
# -----------------------------
def answer_query(
    question: str,
    project_id: Optional[int] = None,
    *,
    k: int = 4
) -> str:

    try:
        docs = search_docs(
            query=question,
            k=k,
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

# -----------------------------
# Debug local
# -----------------------------
if __name__ == "__main__":
    try:
        print("Documents dans la collection :", collection.count())
    except Exception as e:
        print("Erreur Chroma :", e)
