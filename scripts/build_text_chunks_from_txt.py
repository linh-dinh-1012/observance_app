import sqlite3
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "database" / "observance.db"
TXT_DIR  = BASE_DIR / "data" / "avis_couples_txt"   # Folder with TXT avis

# --- Chunking config ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Clear old chunks (optional, comment out if you want to append)
    cur.execute("DELETE FROM text_chunks;")
    cur.execute("DELETE FROM text_chunk_embeddings;")
    conn.commit()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    total_chunks = 0
    txt_files = sorted(TXT_DIR.glob("*.txt"))
    print(f"Dossier TXT: {TXT_DIR} | Fichiers trouvés: {len(txt_files)}")

    for txt_fp in txt_files:
        stem = txt_fp.stem               
        pdf_name = stem + ".pdf"         

        # Lookup file_id in table files
        row = cur.execute(
            "SELECT id FROM files WHERE file_name = ? LIMIT 1", 
            (pdf_name,)
        ).fetchone()
        if not row:
            print(f"Aucun file_id pour '{txt_fp.name}' (attendu '{pdf_name}')")
            continue
        file_id = row[0]

        # Lookup project_id via project_files
        prow = cur.execute(
            "SELECT project_id FROM project_files WHERE file_id = ? LIMIT 1", 
            (file_id,)
        ).fetchone()
        if not prow:
            print(f"Pas de project_id trouvé pour file_id={file_id}")
            continue
        project_id = prow[0]

        # Load text
        text = txt_fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"Fichier vide: {txt_fp.name} — ignoré")
            continue

        # Split into chunks
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            cur.execute("""
                INSERT INTO text_chunks (project_id, file_id, chunk_index, text)
                VALUES (?, ?, ?, ?)
            """, (project_id, file_id, i, chunk))
        total_chunks += len(chunks)
        print(f"{txt_fp.name}: {len(chunks)} chunks (project_id={project_id}, file_id={file_id})")

    conn.commit()
    conn.close()
    print(f"Terminé. Total chunks insérés: {total_chunks}")

if __name__ == "__main__":
    main()
