from pathlib import Path
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

TXT_DIR = Path("demo_docs")
CHROMA_PATH = "chroma_store"
COLLECTION = "demo_docs"

# OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma
client = PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(name=COLLECTION)

# Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

docs, metas, ids = [], [], []
i = 0

for f in TXT_DIR.glob("*.txt"):
    text = f.read_text(encoding="utf-8")
    project_id = int(f.stem)

    for chunk in splitter.split_text(text):
        docs.append(chunk)
        metas.append({
            "project_id": project_id,
            "source": f.name
        })
        ids.append(f"{project_id}_{i}")
        i += 1


col.add(documents=docs, metadatas=metas, ids=ids)
print("Embedding finished")
