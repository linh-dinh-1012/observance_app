import os
import sqlite3
from typing import Iterable, List, Optional, Tuple, Dict, Any

DB_FILE = "database/observance.db"

# ------------------------------
# Connection helpers
# ------------------------------
def get_connection() -> sqlite3.Connection:
    """Open a connection and enable foreign key checks. Row factory returns dict-like rows."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------
# Files
# ------------------------------
def insert_file(
    file_name: str,
    file_type: str,
    file_path: Optional[str],
    file_hash: str,
    date_publication: Optional[str] = None,
    nb_pages: Optional[int] = None,
    **kwargs,
) -> None:
    """Insert one file into the files table (ignores if duplicate hash)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO files
              (file_name, file_type, file_path, file_hash, date_publication, nb_pages)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_name, file_type, file_path, file_hash, date_publication, nb_pages),
        )

def get_file_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM files WHERE file_hash = ?", (file_hash,)).fetchone()
        return dict(row) if row else None

def list_all_files(limit: int = 10) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, file_name, file_type, date_publication FROM files ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

def delete_files_by_id(file_id: int) -> None:
    """Cascade-deletes recommendations via FK. Chunks are not linked to files and are unaffected."""
    with get_connection() as conn:
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

# ------------------------------
# Couples (avis <-> reponse)
# ------------------------------
def link_avis_reponse(project_id: int, avis_id: int, reponse_id: int) -> None:
    """
    Create an (avis, reponse) pair for a given project.

    Constraints (enforced by DB triggers):
    - `avis_id` must reference a file with file_type='avis'
    - `reponse_id` must reference a file with file_type='reponse'
    - both files must belong to `project_id` (exist in `project_files`)
    """
    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO couples (project_id, avis_id, reponse_id)
                VALUES (?, ?, ?)
                """,
                (project_id, avis_id, reponse_id),
            )
        except sqlite3.IntegrityError as e:
            # Raise a clearer message for callers
            raise ValueError(f"Failed to create avis–reponse pair: {e}") from e


# ------------------------------
# Extracted texts
# ------------------------------
def insert_extracted_texts(
    avis_file_id: int,
    reponse_id: Optional[int] = None,        # backward-compat
    synthese: Optional[str] = None,
    avis_complet: Optional[str] = None,
    response: Optional[str] = None,
) -> None:
    """
    Insert structured extracted text into 'extracted_texts'.
    Schema: (file_id, content, synthese, conclusion)
    """
    content = avis_complet if avis_complet is not None else response
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO extracted_texts (file_id, content, synthese, conclusion)
            VALUES (?, ?, ?, ?)
            """,
            (avis_file_id, content, synthese, None),
        )

def get_texts_by_avis_id(avis_file_id: int) -> Optional[Dict[str, Optional[str]]]:
    """Get latest extracted sections by file_id (formerly 'avis_id')."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT content, synthese, conclusion
            FROM extracted_texts
            WHERE file_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (avis_file_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "synthese": row["synthese"],
            "avis_complet": row["content"],
            "recommandations": None,
            "response": None,
        }
    
# ------------------------------
# Recommendations (3 cols)
# ------------------------------
def insert_recommendation(
    file_id: int, recommandation_index: Optional[int], recommandation_text: str
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recommendations (file_id, recommandation_index, recommandation_text)
            VALUES (?, ?, ?)
            """,
            (file_id, recommandation_index, recommandation_text),
        )

def bulk_insert_recommendations(
    rows: Iterable[Tuple[int, Optional[int], str]]
) -> int:
    """rows: iterable of (file_id, recommandation_index, recommandation_text)."""
    rows = list(rows)
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO recommendations (file_id, recommandation_index, recommandation_text)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        return conn.total_changes

def get_recommendations_by_file(file_id: int, order: bool = True) -> List[Dict[str, Any]]:
    sql = "SELECT id, file_id, recommandation_index, recommandation_text FROM recommendations WHERE file_id = ?"
    if order:
        sql += " ORDER BY recommandation_index"
    with get_connection() as conn:
        rows = conn.execute(sql, (file_id,)).fetchall()
        return [dict(r) for r in rows]

def search_recommendations_fts(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """FTS5 search on recommendations.recommandation_text."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.file_id, r.recommandation_index, r.recommandation_text
            FROM recommendations r
            JOIN recommendations_fts fts ON fts.rowid = r.id
            WHERE fts MATCH ?
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

# ------------------------------
# Text chunks (for RAG)
# ------------------------------
def insert_text_chunk(
    file_name: Optional[str],
    project_id: Optional[int],
    chunk_index: Optional[int],
    section: Optional[str],
    text: str,
    topics: Optional[str] = None,
    numbers: Optional[str] = None,
    tokens_est: Optional[int] = None,
    words: Optional[int] = None,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO text_chunks
              (file_name, project_id, chunk_index, section, text, topics, numbers, tokens_est, words)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (file_name, project_id, chunk_index, section, text, topics, numbers, tokens_est, words),
        )
        return cur.lastrowid

def bulk_insert_text_chunks(rows: Iterable[Dict[str, Any]]) -> int:
    """rows: iterable of dicts with keys matching text_chunks columns; required key: 'text'."""
    rows = list(rows)
    if not rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO text_chunks
              (file_name, project_id, chunk_index, section, text, topics, numbers, tokens_est, words)
            VALUES (:file_name, :project_id, :chunk_index, :section, :text, :topics, :numbers, :tokens_est, :words)
            """,
            rows,
        )
        return conn.total_changes

def bulk_insert_text_chunks_from_csv(csv_path: str) -> int:
    """Load chunks from CSV and insert into text_chunks (keeps only known columns)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    keep = ["file_name", "project_id", "chunk_index", "section", "text", "topics", "numbers", "tokens_est", "words"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    rows = df[keep].to_dict(orient="records")
    return bulk_insert_text_chunks(rows)

def search_chunks_fts(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """FTS5 search on text_chunks.text."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT tc.id, tc.file_name, tc.project_id, tc.chunk_index, tc.section, tc.text
            FROM text_chunks tc
            JOIN text_chunks_fts fts ON fts.rowid = tc.id
            WHERE fts MATCH ?
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]
