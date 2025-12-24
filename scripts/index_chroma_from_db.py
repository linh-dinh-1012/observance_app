import sqlite3
from pathlib import Path
from tqdm.auto import tqdm

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# ----------------- Config -----------------
BASE_DIR     = Path(__file__).resolve().parent.parent
DB_PATH      = BASE_DIR / "database" / "observance.db"
PERSIST_DIR  = BASE_DIR / "chroma_store"
COLLECTION   = "gouvernance"
MODEL_NAME   = "intfloat/multilingual-e5-base"

UPSERT_BATCH = 4000   # must be <= 5461 (Chroma limit)

# -------------- Device detect -------------
def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

device = detect_device()
print(f"🔧 Utilisation du device: {device}")

# Load data from DB 
print("Chargement des chunks depuis la base SQLite...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

rows = cur.execute("""
    SELECT id, project_id, file_id, chunk_index, text
    FROM text_chunks
    WHERE text IS NOT NULL AND TRIM(text) != ''
""").fetchall()
conn.close()

print(f"{len(rows)} chunks trouvés dans la base")

# Prepare embedding 
# This is the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# Open Chroma store 
# Persistent client ensures vectors are stored in chroma_store
client = PersistentClient(path=str(PERSIST_DIR))

# Clear collection before re-indexing
try:
    client.delete_collection(COLLECTION)
    print(f"Ancienne collection '{COLLECTION}' supprimée.")
except Exception:
    pass

client.create_collection(COLLECTION)
print(f"Nouvelle collection '{COLLECTION}' créée.")

vectorstore = Chroma(
    collection_name=COLLECTION,
    client=client,
    embedding_function=embeddings
)


# Step 4: Prepare data 
ids = []
texts = []
metas = []

for row in rows:
    chunk_id, project_id, file_id, chunk_index, text = row
    ids.append(f"chunk_{chunk_id}")  # unique ID
    texts.append(text)
    metas.append({
        "project_id": str(project_id) if project_id else "",
        "file_id": str(file_id) if file_id else "",
        "chunk_index": str(chunk_index)
    })

# Embedding + Indexing 
print("Début de l'embedding et de l'indexation dans Chroma...")

# NOTE:
# - Embedding = converting text -> vector (HuggingFace model)
# - Indexing  = adding vectors + metadata into Chroma collection

for s in tqdm(range(0, len(texts), UPSERT_BATCH), desc="Upserting par batch"):
    e = s + UPSERT_BATCH
    vectorstore.add_texts(
        texts=texts[s:e],
        metadatas=metas[s:e],
        ids=ids[s:e]
    )

print(f"Terminé. {len(texts)} chunks ajoutés à Chroma.")
print(f"Modèle = {MODEL_NAME} | Device = {device} | Collection = '{COLLECTION}'")
