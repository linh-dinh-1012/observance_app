import os
import sqlite3

DB_FILE = os.path.join(os.path.dirname(__file__), "observance.db")

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")  
    c = conn.cursor()
    print("DB path:", os.path.abspath(DB_FILE))

    # Files table
    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_type TEXT,                -- 'avis'/'reponse'/'other'
            file_path TEXT,
            file_hash TEXT UNIQUE,
            nb_pages INTEGER,
            date_publication TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Projects table 
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            titre TEXT,
            descriptif TEXT,
            nature TEXT,
            localisation TEXT,
            departement TEXT,
            nb_recommandations INTEGER,
            avis_critique TEXT,                     -- Peu critique / Mitigé / Assez critique à critique / Très critique
            citation_recom TEXT,                    -- Cite toutes recom explicitement + reponse / Ne cite pas explicitement + reponse / Cite certaine + reponse
            signataire_reponse TEXT,
            conclusion_reponse TEXT, 
            maitre_ouvrage TEXT,
            bureau_etude TEXT
        )
    ''')

    # Project_files (one-to-many)
    c.execute(''' 
        CREATE TABLE IF NOT EXISTS project_files (
            project_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            role TEXT CHECK (role IN ('avis', 'reponse', 'other')),
            PRIMARY KEY (project_id, file_id),
            FOREIGN KEY (project_id) REFERENCES projects(id),
            FOREIGN KEY (file_id) REFERENCES files(id)
        )
    ''')

    # Extracted_texts
    c.execute('''
        CREATE TABLE IF NOT EXISTS extracted_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            content TEXT,
            synthese TEXT,
            avis_detaille TEXT,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE RESTRICT
        )
    ''')

    # Recommendations 
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            recommandation_index INTEGER,
            recommandation_text TEXT NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_recom_file ON recommendations(file_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_recom_order ON recommendations(file_id, recommandation_index)")

    # Couples (avis <-> reponse) 
    c.execute('''
        CREATE TABLE IF NOT EXISTS couples (
            project_id  INTEGER NOT NULL,
            avis_id     INTEGER NOT NULL,
            reponse_id  INTEGER NOT NULL,
            CHECK (avis_id <> reponse_id),
            PRIMARY KEY (project_id, avis_id, reponse_id),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (avis_id)    REFERENCES files(id)    ON DELETE CASCADE,
            FOREIGN KEY (reponse_id) REFERENCES files(id)    ON DELETE CASCADE
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_couples_project ON couples(project_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_couples_avis    ON couples(avis_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_couples_reponse ON couples(reponse_id)")

    # Trigger: ensure correct file type & same project_id (based on project_files)
    c.execute('''
        CREATE TRIGGER IF NOT EXISTS couples_bi BEFORE INSERT ON couples
        BEGIN
            -- Check file_type
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM files WHERE id=new.avis_id AND file_type='avis')
                THEN RAISE(ABORT, 'avis_id is not file_type=avis')
            END;
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM files WHERE id=new.reponse_id AND file_type='reponse')
                THEN RAISE(ABORT, 'reponse_id is not file_type=reponse')
            END;
            -- Check for the same project_id
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM project_files WHERE project_id=new.project_id AND file_id=new.avis_id)
                THEN RAISE(ABORT, 'avis_id does not belong to the given project_id')
            END;
            SELECT CASE
                WHEN NOT EXISTS (SELECT 1 FROM project_files WHERE project_id=new.project_id AND file_id=new.reponse_id)
                THEN RAISE(ABORT, 'reponse_id does not belong to the given project_id')
            END;
        END;
    ''')

    # Thematiques
    c.execute('''
        CREATE TABLE IF NOT EXISTS thematiques (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thematique TEXT UNIQUE
        )
    ''')

    # Project_thematiques (many-to-many)
    c.execute('''
        CREATE TABLE IF NOT EXISTS project_thematiques (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            thematique_id INTEGER,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (thematique_id) REFERENCES thematiques(id) ON DELETE CASCADE
        )
    ''')

    # Text_chunks table (for RAG)
    c.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            chunk_index INTEGER,
            section TEXT,
            text TEXT NOT NULL,
            topics TEXT,
            numbers TEXT,
            tokens_est INTEGER,
            words INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_id INTEGER REFERENCES files(id),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
        )
    ''')

    # Indexes for faster search
    c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON text_chunks(project_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON text_chunks(file_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_order ON text_chunks(file_id, chunk_index)")

    # Embeddings table 
    c.execute('''
        CREATE TABLE IF NOT EXISTS text_chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chunk_id) REFERENCES text_chunks(id) ON DELETE CASCADE
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_embed_model ON text_chunk_embeddings(model)")

    # FTS5 virtual table for keyword search - CHUNKS
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks_fts
        USING fts5(text, content='text_chunks', content_rowid='id');
    ''')

    # Triggers to sync FTS - CHUNKS
    c.execute('''
        CREATE TRIGGER IF NOT EXISTS text_chunks_ai AFTER INSERT ON text_chunks
        BEGIN
            INSERT INTO text_chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
    ''')
    c.execute('''
        CREATE TRIGGER IF NOT EXISTS text_chunks_ad AFTER DELETE ON text_chunks
        BEGIN
            INSERT INTO text_chunks_fts(text_chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;
    ''')
    c.execute('''
        CREATE TRIGGER IF NOT EXISTS text_chunks_au AFTER UPDATE ON text_chunks
        BEGIN
            INSERT INTO text_chunks_fts(text_chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO text_chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
    ''')

    # FTS5 for RECOMMENDATIONS
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS recommendations_fts
        USING fts5(recommandation_text, content='recommendations', content_rowid='id');
    ''')

    c.execute('''
        CREATE TRIGGER IF NOT EXISTS recommendations_ai AFTER INSERT ON recommendations
        BEGIN
            INSERT INTO recommendations_fts(rowid, recommandation_text)
            VALUES (new.id, new.recommandation_text);
        END;
    ''')

    c.execute('''
        CREATE TRIGGER IF NOT EXISTS recommendations_ad AFTER DELETE ON recommendations
        BEGIN
            INSERT INTO recommendations_fts(recommendations_fts, rowid, recommandation_text)
            VALUES('delete', old.id, old.recommandation_text);
        END;
    ''')

    c.execute('''
        CREATE TRIGGER IF NOT EXISTS recommendations_au AFTER UPDATE ON recommendations
        BEGIN
            INSERT INTO recommendations_fts(recommendations_fts, rowid, recommandation_text)
            VALUES('delete', old.id, old.recommandation_text);
            INSERT INTO recommendations_fts(rowid, recommandation_text)
            VALUES (new.id, new.recommandation_text);
        END;
    ''')

    conn.commit()
    conn.close()
    print(f"La base de données '{DB_FILE}' a été créée avec succès avec tables RAG + FTS pour recommendations!")

if __name__ == "__main__":
    init_db()
