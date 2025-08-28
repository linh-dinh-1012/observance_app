import os
import sqlite3
import pandas as pd

# === Paths ===
# Place this script next to `init_database.py`.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.normpath(os.path.join(BASE_DIR, "../data/tables"))
DB_FILE = os.path.join(BASE_DIR, "observance.db")  # must match init_database.py


def normalize_file_hash(series: pd.Series) -> pd.Series:
    """
    Normalize file_hash values:
    - strip whitespace
    - convert empty strings to NA/None so they won't collide on UNIQUE(file_hash)
    """
    series = series.apply(lambda x: x.strip() if isinstance(x, str) else x)
    series = series.replace("", pd.NA)
    return series


def import_csv_to_db():
    # Ensure DB exists and enforce foreign keys
    if not os.path.exists(DB_FILE):
        # Optional: auto-create schema if `init_database.py` is available
        try:
            from init_database import init_db
            print("DB not found; creating schema with init_database.init_db()...")
            init_db()
        except Exception as e:
            raise SystemExit(
                f"Database not found at {DB_FILE} and failed to auto-create.\n"
                f"Run: python3 init_database.py\nOriginal error: {e}"
            )

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    print("Using DB file:", os.path.abspath(DB_FILE))

    # -------------------------------------------------------
    # OPTIONAL: uncomment to reset tables before import
    # (useful for a clean re-import)
    #
    # conn.execute("DELETE FROM text_chunk_embeddings")
    # conn.execute("DELETE FROM text_chunks")
    # conn.execute("DELETE FROM project_thematiques")
    # conn.execute("DELETE FROM thematiques")
    # conn.execute("DELETE FROM recommendations")
    # conn.execute("DELETE FROM extracted_texts")
    # conn.execute("DELETE FROM couples")
    # conn.execute("DELETE FROM project_files")
    # conn.execute("DELETE FROM projects")
    # conn.execute("DELETE FROM files")
    # conn.commit()
    # -------------------------------------------------------

    # ===== 1) FILES =====
    files_path = os.path.join(CSV_DIR, "files_table.csv")
    files_df = pd.read_csv(files_path)

    # Keep only schema columns
    files_keep = ["file_name", "file_type", "file_path", "file_hash", "nb_pages", "date_publication"]
    for col in files_keep:
        if col not in files_df.columns:
            files_df[col] = None

    # Clean file_hash for deduplication
    if "file_hash" in files_df.columns:
        files_df["file_hash"] = normalize_file_hash(files_df["file_hash"])
        with_hash = files_df[files_df["file_hash"].notna()].copy()
        without_hash = files_df[files_df["file_hash"].isna()].copy()
        with_hash = with_hash.drop_duplicates(subset=["file_hash"], keep="first")
        files_df = pd.concat([with_hash, without_hash], ignore_index=True)

    # Insert into files
    rows = files_df[[c for c in files_keep if c != "id"]].to_dict(orient="records")
    conn.executemany(
        """
        INSERT OR IGNORE INTO files
          (file_name, file_type, file_path, file_hash, nb_pages, date_publication)
        VALUES (:file_name, :file_type, :file_path, :file_hash, :nb_pages, :date_publication)
        """,
        rows,
    )
    print(f"files: inserted/kept {conn.total_changes} rows (duplicates ignored)")

    # ===== 2) PROJECTS =====
    projects_path = os.path.join(CSV_DIR, "projects_table.csv")
    if os.path.exists(projects_path):
        projects_df = pd.read_csv(projects_path)

        # Map project_id column to id if present
        if "id" in projects_df.columns:
            projects_df = projects_df.drop(columns=["id"])

        # Clear existing projects before import 
        conn.execute("DELETE FROM projects")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='projects'")
        conn.commit()

        projects_df.to_sql("projects", conn, if_exists="append", index=False)
        print("projects: imported")

    # ===== 3) PROJECT_FILES (link files <-> projects) =====
    if "project_id" in files_df.columns:
        cur = conn.execute("SELECT id, file_name, file_type FROM files")
        files_in_db = {row[1]: (row[0], row[2]) for row in cur.fetchall()}  # file_name -> (id, file_type)

        project_file_rows = []
        for _, row in files_df.iterrows():
            if pd.isna(row.get("project_id")):
                continue
            fname = row["file_name"]
            if fname not in files_in_db:
                continue
            file_id, ftype = files_in_db[fname]
            project_id = int(row["project_id"])
            role = row["file_type"] if row["file_type"] in ("avis", "reponse", "other") else "other"
            project_file_rows.append((project_id, file_id, role))

        conn.executemany(
            "INSERT OR IGNORE INTO project_files (project_id, file_id, role) VALUES (?, ?, ?)",
            project_file_rows
        )
        print(f"project_files: linked {len(project_file_rows)} files to projects")

    # ===== 4) COUPLES (avis <-> reponse, optional) =====
    if "project_id" in files_df.columns:
        for pid in files_df["project_id"].dropna().unique():
            avis = files_df[(files_df["project_id"] == pid) & (files_df["file_type"] == "avis")]
            rep  = files_df[(files_df["project_id"] == pid) & (files_df["file_type"] == "reponse")]
            if not avis.empty and not rep.empty:
                avis_name = avis.iloc[0]["file_name"]
                rep_name  = rep.iloc[0]["file_name"]
                if avis_name in files_in_db and rep_name in files_in_db:
                    avis_id = files_in_db[avis_name][0]
                    rep_id  = files_in_db[rep_name][0]
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO couples (project_id, avis_id, reponse_id) VALUES (?, ?, ?)",
                            (int(pid), avis_id, rep_id)
                        )
                    except Exception as e:
                        print(f"⚠️ Skip couple for project {pid}: {e}")
        print("couples: linked avis<->reponse where possible")

    # ===== 5) EXTRACTED_TEXTS =====
    extracted_path = os.path.join(CSV_DIR, "extracted_texts.csv")
    if os.path.exists(extracted_path):
        extracted_df = pd.read_csv(extracted_path)
        extracted_df.to_sql("extracted_texts", conn, if_exists="append", index=False)
        print("extracted_texts: imported")

    # ===== 6) RECOMMENDATIONS =====
    recom_path = os.path.join(CSV_DIR, "recommendations_table.csv")
    if os.path.exists(recom_path):
        recom_df = pd.read_csv(recom_path)
        recom_df.to_sql("recommendations", conn, if_exists="append", index=False)
        print("recommendations: imported")

    # ===== 7) THEMATIQUES =====
    thema_path = os.path.join(CSV_DIR, "thematiques_table.csv")
    if os.path.exists(thema_path):
        thema_df = pd.read_csv(thema_path)
        thema_df.to_sql("thematiques", conn, if_exists="append", index=False)
        print("thematiques: imported")

    # ===== 8) TEXT_CHUNKS =====
    chunks_path = os.path.join(CSV_DIR, "text_chunks.csv")
    if os.path.exists(chunks_path):
        chunks_df = pd.read_csv(chunks_path)

        # Map file_name -> file_id if needed
        cur = conn.execute("SELECT id AS file_id, file_name FROM files")
        name_to_id = {row[1]: row[0] for row in cur.fetchall()}
        if "file_id" not in chunks_df.columns and "file_name" in chunks_df.columns:
            chunks_df["file_id"] = chunks_df["file_name"].map(name_to_id)

        chunks_keep = [
            "project_id", "file_id", "chunk_index", "section", "text",
            "topics", "numbers", "tokens_est", "words"
        ]
        for col in chunks_keep:
            if col not in chunks_df.columns:
                chunks_df[col] = None

        chunks_df = chunks_df[chunks_df["text"].notna() & (chunks_df["text"].astype(str).str.strip() != "")]
        chunks_df[chunks_keep].to_sql("text_chunks", conn, if_exists="append", index=False)
        print("text_chunks: imported")

    # ===== 9) TEXT_CHUNK_EMBEDDINGS =====
    embeds_path = os.path.join(CSV_DIR, "text_chunk_embeddings.csv")
    if os.path.exists(embeds_path):
        embeds_df = pd.read_csv(embeds_path)
        embeds_keep = ["chunk_id", "model", "dim", "vector_json", "created_at"]
        for col in embeds_keep:
            if col not in embeds_df.columns:
                embeds_df[col] = None
        embeds_df[embeds_keep].to_sql("text_chunk_embeddings", conn, if_exists="append", index=False)
        print("text_chunk_embeddings: imported")

    # ===== 10) FTS5 sync =====
    conn.execute("DELETE FROM text_chunks_fts")
    conn.execute("INSERT INTO text_chunks_fts(rowid, text) SELECT id, text FROM text_chunks")
    print("✅ text_chunks_fts synced")

    conn.execute("DELETE FROM recommendations_fts")
    conn.execute("INSERT INTO recommendations_fts(rowid, recommandation_text) SELECT id, recommandation_text FROM recommendations")
    print("✅ recommendations_fts synced")

    conn.commit()
    conn.close()
    print("All CSVs imported into database!")


if __name__ == "__main__":
    import_csv_to_db()
