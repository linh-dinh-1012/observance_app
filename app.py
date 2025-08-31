import streamlit as st
import time
import sqlite3
import pandas as pd
import os, requests, zipfile 
from chromadb import PersistentClient
import plotly.express as px
from shapely import wkt
from shapely.geometry import mapping
from pathlib import Path
from streamlit_option_menu import option_menu # type: ignore
from streamlit_tags import st_tags # type: ignore
from scripts.rag import search_docs, generate_answer_stream  # Import RAG function
from google.cloud import storage

# =========================================================
# THEME & UI HELPERS
# =========================================================
PRIMARY = "#0f4336"       # dark green sidebar
PRIMARY_600 = "#145a46"
BORDER = "#e5e7eb"
RADIUS = "16px"
SHADOW = "0 4px 18px rgba(0,0,0,.06)"

def inject_css():
    st.markdown(f"""
    <style>
    /* ====== Page container ====== */
    .main .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }}

    /* ====== Sidebar color ====== */
    section[data-testid="stSidebar"] {{
        background: {PRIMARY};
        color: #fff;
        border-right: 1px solid rgba(255,255,255,.08);
    }}
    section[data-testid="stSidebar"] * {{ color:#fff !important; }}

    /* ====== Cards ====== */
    .card {{
        background:#fff; border:1px solid {BORDER}; border-radius:{RADIUS};
        box-shadow:{SHADOW}; padding:16px 18px;
    }}
    .card + .card {{ margin-top: 14px; }}
    .card-title {{ font-weight:700; font-size:1.05rem; margin-bottom:10px; color:#1f2937; }}

    /* KPI cards */
    .kpi {{
        background:#fff; border:1px solid {BORDER}; border-radius:{RADIUS};
        box-shadow:{SHADOW}; padding:18px 22px; text-align:center;
    }}
    .kpi .label {{ color:#374151; font-weight:600; }}
    .kpi .value {{ font-size:2rem; font-weight:800; line-height:1; margin-top:6px; }}

    /* Default st.metric widget */
    div[data-testid="stMetric"] {{
        border:1px solid {BORDER}; border-radius:{RADIUS}; padding:16px 18px;
        background:#fff; box-shadow:{SHADOW};
    }}

    /* Buttons */
    .stButton > button {{
        background:{PRIMARY}; color:white; border:0; padding:.7rem 1.1rem;
        border-radius:12px; font-weight:700;
    }}
    .stButton > button:hover {{ background:{PRIMARY_600}; }}

    /* Tables */
    .stTable {{ border-radius:{RADIUS}; overflow:hidden; box-shadow:{SHADOW}; }}
    </style>
    """, unsafe_allow_html=True)

def kpi_card(label: str, value):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="value">{value:,}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def card_open(title: str = ""):
    st.markdown(
        f"""<div class="card">{f'<div class="card-title">{title}</div>' if title else ''}""",
        unsafe_allow_html=True
    )

def card_close():
    st.markdown("""</div>""", unsafe_allow_html=True)

# =========================================================
# CONFIG: Links to Seafile data
# =========================================================

# DB_URL = "https://seafile.agroparistech.fr/seafhttp/f/8dcbc28bbe8f441090a2/?op=view"
# CHROMA_URL = "https://seafile.agroparistech.fr/seafhttp/f/014807ad29f34fbe9128/?op=view"


# Google Cloud Storage bucket name
BUCKET_NAME = "observance-app-2025"
DB_BLOB = "database/observance.db"
CHROMA_BLOB = "chroma_store.zip"

# =========================================================
# DATABASE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"   # downloaded DB will be saved here
CHROMA_PATH = BASE_DIR / "chroma_store"  # extracted Chroma store folder


def download_from_gcs(bucket_name, blob_name, out_path, label):
    """Download a file from GCS to local path with progress info in Streamlit"""
    st.info(f"⬇️ Downloading {label} from GCS...")
    t0 = time.time()

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    dt = time.time() - t0
    st.success(f"{label} downloaded ✅ ({size_mb:.1f} MB in {dt:.1f}s)")


@st.cache_resource
def setup_data():
    """Ensure database + Chroma store are available locally.
    If not found, download from GCS.
    """
    # ---- Ensure database ----
    if not DB_PATH.exists():
        download_from_gcs(BUCKET_NAME, DB_BLOB, DB_PATH, "Database")

    # ---- Ensure Chroma store ----
    if not CHROMA_PATH.exists():
        zip_path = BASE_DIR / "chroma_store.zip"
        download_from_gcs(BUCKET_NAME, CHROMA_BLOB, zip_path, "Chroma store (zip)")
        st.info("Extracting Chroma store...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CHROMA_PATH)
        st.success("Chroma store extracted ✅")

    # ---- Return DB connection + Chroma client ----
    conn = sqlite3.connect(DB_PATH)
    client = PersistentClient(path=str(CHROMA_PATH))
    return conn, client

# @st.cache_resource
# def setup_data():
#     """Ensure database + Chroma store are available locally.
#     If not found, download from Seafile (direct download link).
#     Show progress bar + size + time in Streamlit.
#     """

#     def download_with_progress(url, out_path, label):
#         """Download file with progress bar and return size + time"""
#         st.info(f"⬇️ Downloading {label}...")
#         t0 = time.time()
#         r = requests.get(url, stream=True)
#         r.raise_for_status()
#         total_size = int(r.headers.get("content-length", 0))
#         block_size = 1024 * 1024  # 1 MB
#         progress_bar = st.progress(0, text=f"Downloading {label}...")
#         size_dl = 0
#         with open(out_path, "wb") as f:
#             for data in r.iter_content(block_size):
#                 f.write(data)
#                 size_dl += len(data)
#                 if total_size:
#                     progress = min(size_dl / total_size, 1.0)
#                     progress_bar.progress(progress, text=f"{label} {progress*100:.1f}%")
#         dt = time.time() - t0
#         size_mb = size_dl / (1024 * 1024)
#         progress_bar.empty()
#         st.success(f"{label} downloaded ✅ ({size_mb:.1f} MB in {dt:.1f}s)")

#     # ---- Ensure database ----
#     if not DB_PATH.exists():
#         os.makedirs(DB_PATH.parent, exist_ok=True)
#         download_with_progress(DB_URL, DB_PATH, "Database")

#     # ---- Ensure Chroma store ----
#     if not CHROMA_PATH.exists():
#         os.makedirs(CHROMA_PATH.parent, exist_ok=True)
#         zip_path = BASE_DIR / "chroma_store.zip"
#         download_with_progress(CHROMA_URL, zip_path, "Chroma store (zip)")
#         st.info("Extracting Chroma store...")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(CHROMA_PATH)
#         st.success("Chroma store extracted ✅")

#     # ---- Return DB connection + Chroma client ----
#     conn = sqlite3.connect(DB_PATH)
#     client = PersistentClient(path=str(CHROMA_PATH))
#     return conn, client

def run_query(query, params=()):
    """Execute a read-only SQL query and return a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df

# =========================================================
# TAB 1: Dashboard
# =========================================================
def dashboard():
    st.markdown("## Tableau de bord")

    # --- Counts for Avis & Réponses ---
    df_counts = run_query("""
        SELECT file_type, COUNT(*) as count
        FROM files
        GROUP BY file_type
    """)
    avis_count = int(df_counts.loc[df_counts['file_type'] == 'avis', 'count'].sum()) if not df_counts.empty else 0
    reponse_count = int(df_counts.loc[df_counts['file_type'] == 'reponse', 'count'].sum()) if not df_counts.empty else 0

    # KPI row
    k1, k2 = st.columns([1,1])
    with k1: kpi_card("Avis", avis_count)
    with k2: kpi_card("Réponses", reponse_count)
    
    card_close()

    # --- Evolution of avis over time (line chart) ---
    left, right = st.columns([1.35, 1])
    with left:
        card_open("Évolution des avis")
        df_dates = run_query("""
            SELECT date_publication
            FROM files
            WHERE file_type = 'avis' AND date_publication IS NOT NULL
        """)
        if df_dates.empty:
            st.info("No dates available for avis.")
        else:
            s = df_dates["date_publication"].astype(str).str.strip().str.slice(0, 10)
            years = pd.to_datetime(s, errors="coerce").dt.year.dropna().astype(int)
            df_year = (
                years.value_counts()
                    .sort_index()
                    .rename_axis("year")
                    .reset_index(name="count")
            )
            if df_year.empty:
                st.info("No valid dates found to plot evolution.")
            else:
                df_year["year"] = df_year["year"].astype(str)
                fig_year = px.line(df_year, x="year", y="count", markers=True)
                fig_year.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                fig_year.update_xaxes(type="category")
                st.plotly_chart(fig_year, use_container_width=True, key="evolution_avis")
        card_close()

    # --- Bar chart: Distribution of critique levels ---
    with right:
        card_open("Répartition du niveau critique des avis")
        df_crit = run_query("""
            SELECT avis_critique, COUNT(*) as count
            FROM projects
            GROUP BY avis_critique
        """)
        if df_crit.empty:
            st.info("No data.")
        else:
            df_crit["avis_code"] = pd.to_numeric(df_crit["avis_critique"], errors="coerce").round().astype("Int64")
            labels_map = {
                1: "1. Peu critique",
                2: "2. Mitigé",
                3: "3. Assez critique à critique",
                4: "4. Très critique",
            }
            df_crit["avis_label"] = df_crit["avis_code"].map(labels_map).fillna("Non renseigné")
            df_plot = df_crit.groupby("avis_label", as_index=False)["count"].sum()
            order = [
                "1. Peu critique","2. Mitigé","3. Assez critique à critique","4. Très critique","Non renseigné"
            ]
            df_plot["avis_label"] = pd.Categorical(df_plot["avis_label"], categories=order, ordered=True)
            df_plot = df_plot.sort_values("avis_label")

            blues = px.colors.sequential.Blues[1:5]
            color_map = {
                "1. Peu critique": blues[0],
                "2. Mitigé": blues[1],
                "3. Assez critique à critique": blues[2],
                "4. Très critique": blues[3],
                "Non renseigné": "#d9d9d9",
            }
            fig_bar = px.bar(
                df_plot, x="avis_label", y="count",
                color="avis_label", color_discrete_map=color_map, text="count"
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                xaxis_title=None, yaxis_title=None, showlegend=False,
                bargap=0.2, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_avis")
        card_close()

    # --- Top 5 critical projects ---
    card_open("Top 5 des projets critiqués")
    df_top = run_query("""
        SELECT titre, nb_recommandations, avis_critique
        FROM projects
        ORDER BY CAST(avis_critique AS INTEGER) DESC
        LIMIT 5
    """)
    st.table(df_top)
    card_close()

    # --- Project map ---
    card_open("Emplacements des projets")
    def map_projects():
        df = run_query("SELECT id, titre, localisation, nb_recommandations, avis_critique FROM projects")
        features = []
        for _, row in df.iterrows():
            loc = row["localisation"]
            if pd.isna(loc) or not str(loc).strip():
                continue
            try:
                geom = wkt.loads(loc)
                geojson_obj = mapping(geom)
                features.append({
                    "type": "Feature",
                    "geometry": geojson_obj,
                    "properties": {
                        "id": row["id"],
                        "titre": row["titre"],
                        "nb_recommandations": row["nb_recommandations"],
                        "avis_critique": row["avis_critique"]
                    }
                })
            except Exception as e:
                print(f"[DEBUG] Error parsing location for project {row['id']}: {e}")
                continue
        if not features:
            st.info("No location data available.")
            return
        geojson_data = {"type": "FeatureCollection", "features": features}
        df_map = pd.DataFrame([f["properties"] for f in features])

        fig_map = px.choropleth_map(
            df_map,
            geojson=geojson_data,
            locations="id",
            color="nb_recommandations",
            hover_name="titre",
            hover_data={"nb_recommandations": True, "avis_critique": True, "id": False},
            map_style="carto-positron",
            center={"lat": 46.6, "lon": 2.2},
            zoom=5,
            opacity=0.7,
            featureidkey="properties.id"
        )
        fig_map.update_traces(marker_line_width=2, marker_line_color="black")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, hoverlabel=dict(font_size=14, font_family="Arial"))
        fig_map.update_geos(fitbounds="locations")
        st.plotly_chart(fig_map, use_container_width=True, key="map_projects")
    map_projects()
    card_close()

# =========================================================
# TAB 2: Search & Analyse
# =========================================================
@st.cache_data
def get_distinct(table: str, col: str, where: str = "1=1") -> list[str]:
    """
    Return a list of distinct, non-NULL values for <table>.<col>.
    `where` lets you restrict by file_type, etc.
    """
    q = f"""
        SELECT DISTINCT {col} AS v
        FROM {table}
        WHERE {where} AND {col} IS NOT NULL
    """
    df = run_query(q)
    return df["v"].dropna().astype(str).tolist()

def pretty_critique(raw: str | None) -> str:
    """
    Map numeric levels 1-4 → human-friendly French labels.
    If the DB already stores text, simply return it.
    """
    if raw is None:
        return "N/A"
    mapping = {
        "1": "1. Peu critique",
        "2": "2. Mitigé",
        "3": "3. Assez critique à critique",
        "4": "4. Très critique",
    }
    s = str(raw).strip()
    return mapping.get(s, s)

def recherche_analyse():
    """Search & Analyse with persistent state and RAG."""
    st.markdown("## 🔍 Recherche & Analyse")

    # ---------- Load distinct values for filters ----------
    titres         = get_distinct("projects", "titre")
    dates_avis     = get_distinct("files", "date_publication", "file_type = 'avis'")
    critiques_raw  = get_distinct("projects", "avis_critique")
    departements   = get_distinct("projects", "departement")

    # ---------- Search form ----------
    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            titre_sel = st.selectbox("Titre du projet", [""] + titres, index=0)
        with col2:
            date_sel = st.selectbox("Date de publication de l'avis", [""] + dates_avis, index=0)
        with col3:
            critique_sel = st.selectbox("Niveau critique de l'avis", [""] + critiques_raw, index=0)

        keywords = st_tags(
            label="Mots-clés",
            text="Entrez un mot-clé et appuyez sur Enter",
            value=[],
            suggestions=sorted(set(titres + dates_avis + departements)),
            maxtags=5,
            key="keyword_search"
        )

        submit_search = st.form_submit_button("Rechercher")

    # ---------- Run SQL search ----------
    if submit_search:
        conditions, params = [], []

        # Filters
        if titre_sel:
            conditions.append("p.titre = ?"); params.append(titre_sel)
        if date_sel:
            conditions.append("fa.date_publication = ?"); params.append(date_sel)
        if critique_sel:
            conditions.append("p.avis_critique = ?"); params.append(critique_sel)

        # Keyword OR-block
        for kw in keywords:
            like = f"%{kw}%"
            conditions.append(
                "(p.titre LIKE ? OR p.nature LIKE ? OR p.departement LIKE ? OR p.avis_critique LIKE ?)"
            )
            params.extend([like, like, like, like])

        if not conditions:
            st.warning("Veuillez sélectionner au moins un critère de recherche.")
        else:
            sql = f"""
                SELECT DISTINCT p.*
                FROM projects p
                LEFT JOIN couples c ON c.project_id = p.id
                LEFT JOIN files   fa ON fa.id = c.avis_id          -- restrict by avis date
                WHERE {" AND ".join(conditions)}
            """
            st.session_state["search_results"] = run_query(sql, params)

            # Clear old selections / answers
            st.session_state.pop("selected_project_label", None)
            for k in list(st.session_state.keys()):
                if str(k).startswith("rag-answer-"):
                    st.session_state.pop(k, None)

    # ---------- Read results from state ----------
    df_results = st.session_state.get("search_results")

    if df_results is None:
        st.info("Saisissez vos critères et cliquez sur **Rechercher**.")
        return
    if df_results.empty:
        st.info("Aucun projet trouvé.")
        return

    st.success(f"{len(df_results)} projets trouvés")

    # ---------- Persist selected project ----------
    options = [f"{int(r.id)} — {r.titre}" for _, r in df_results.iterrows()]
    default_idx = 0
    if (
        "selected_project_label" in st.session_state
        and st.session_state["selected_project_label"] in options
    ):
        default_idx = options.index(st.session_state["selected_project_label"])

    selected_label = st.selectbox(
        "👉 Choisissez un projet pour l'analyse",
        options,
        index=default_idx if options else 0,
        key="selected_project_label"
    )
    if not selected_label:
        st.info("Sélectionnez un projet.")
        return

    selected_id = int(selected_label.split(" — ")[0])
    project = df_results[df_results["id"] == selected_id].iloc[0]

    # ---------- Project card ----------
    card_open(f"📄 Projet sélectionné : {project['titre']}")

    details = run_query(
    """
        SELECT 
            p.titre                           AS titre_projet,
            p.avis_critique                   AS avis_critique_raw,
            p.signataire_reponse              AS signataire,
            p.citation_recom                  AS citation_recom,
            p.conclusion_reponse              AS conclusion,

            -- Avis (from couples or fallback project_files)
            COALESCE(
                (SELECT f.date_publication 
                FROM couples c JOIN files f ON f.id = c.avis_id
                WHERE c.project_id = p.id
                ORDER BY f.date_publication DESC LIMIT 1),
                (SELECT f.date_publication
                FROM project_files pf JOIN files f ON f.id = pf.file_id
                WHERE pf.project_id = p.id AND pf.role='avis'
                ORDER BY f.date_publication DESC LIMIT 1)
            ) AS date_avis,

            COALESCE(
                (SELECT f.nb_pages 
                FROM couples c JOIN files f ON f.id = c.avis_id
                WHERE c.project_id = p.id
                ORDER BY f.date_publication DESC LIMIT 1),
                (SELECT f.nb_pages
                FROM project_files pf JOIN files f ON f.id = pf.file_id
                WHERE pf.project_id = p.id AND pf.role='avis'
                ORDER BY f.date_publication DESC LIMIT 1)
            ) AS pages_avis,

            -- Réponse (from couples or fallback project_files)
            COALESCE(
                (SELECT f.date_publication 
                FROM couples c JOIN files f ON f.id = c.reponse_id
                WHERE c.project_id = p.id
                ORDER BY f.date_publication DESC LIMIT 1),
                (SELECT f.date_publication
                FROM project_files pf JOIN files f ON f.id = pf.file_id
                WHERE pf.project_id = p.id AND pf.role='reponse'
                ORDER BY f.date_publication DESC LIMIT 1)
            ) AS date_reponse,

            COALESCE(
                (SELECT f.nb_pages 
                FROM couples c JOIN files f ON f.id = c.reponse_id
                WHERE c.project_id = p.id
                ORDER BY f.date_publication DESC LIMIT 1),
                (SELECT f.nb_pages
                FROM project_files pf JOIN files f ON f.id = pf.file_id
                WHERE pf.project_id = p.id AND pf.role='reponse'
                ORDER BY f.date_publication DESC LIMIT 1)
            ) AS pages_reponse

        FROM projects p
        WHERE p.id = ?
        """,
        (selected_id,)
    )

    row = details.iloc[0] if not details.empty else None

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return 'N/A'

    # Render the 9 requested fields
    nb_pages_avis = safe_int(row['pages_avis']) if row is not None and pd.notna(row['pages_avis']) else 'N/A'
    nb_pages_rep  = safe_int(row['pages_reponse']) if row is not None and pd.notna(row['pages_reponse']) else 'N/A'
    st.markdown(f"""
        **Titre de projet** : {project.get('titre', 'N/A')}  
        **Date de publication de l'avis** : {row['date_avis'] if row is not None and row['date_avis'] else 'N/A'}  
        **Nombre de page(s) de l'avis** : {nb_pages_avis}  
        **Niveau critique de l'avis** : {pretty_critique(project.get('avis_critique'))}  

        **Date de réponse du maître d'ouvrage** : {row['date_reponse'] if row is not None and row['date_reponse'] else 'N/A'}  
        **Nombre de page(s) de réponse du maître d'ouvrage** : {nb_pages_rep}  
        **La réponse est signée par** : {row['signataire'] if row is not None and row['signataire'] else 'N/A'}  
        **Citation explicite de recommandations de l'Ae** : {project.get('citation_recom', 'N/A')}  
        **Conclusion** : {project.get('conclusion_reponse', 'N/A')}
        """)

    card_close()

    # ---------- RAG / LLM zone ----------
    st.markdown("---")
    st.subheader("💬 Analyse avancée avec RAG")

    answer_key = f"rag-answer-{selected_id}"

    with st.form(f"rag-form-{selected_id}", clear_on_submit=False):
        user_q = st.text_area(
            "Posez votre question sur ce projet",
            key=f"rag-prompt-{selected_id}",
            placeholder="Ex: Résumez les recommandations majeures…"
        )
        run_llm = st.form_submit_button("Analyser avec LLM")

    # Default LLM params
    DEFAULT_K = 8
    DEFAULT_NUM_PREDICT = 256
    DEFAULT_MAX_CTX_CHARS = 12000
    DEFAULT_USE_MMR = False

    if run_llm:
        q = (user_q or "").strip()
        if not q:
            st.warning("Veuillez entrer une question.")
        else:
            with st.spinner("Récupération & génération en cours…"):
                try:
                    # 1) Retrieve docs (MMR on, filtered by project)
                    docs = search_docs(
                        q,
                        k=DEFAULT_K,
                        project_id=selected_id,
                        use_mmr=DEFAULT_USE_MMR
                    )
                    st.caption(f"🔎 Debug: {len(docs)} documents récupérés pour le projet #{selected_id}.")
                    if len(docs) > 0:
                        st.caption(f"Exemple métadonnées: {docs[0].metadata}")

                    # 2) Stream the answer
                    answer = generate_answer_stream(
                        q, docs,
                        num_predict=DEFAULT_NUM_PREDICT,
                        max_context_chars=DEFAULT_MAX_CTX_CHARS,
                    )
                except Exception as e:
                    st.error("Erreur RAG/LLM :")
                    st.exception(e)
                    answer = None

            st.session_state[answer_key] = answer or ""

    # Show last answer after rerun
    if st.session_state.get(answer_key):
        st.markdown("### Réponse :")
        st.write(str(st.session_state[answer_key]))

# =========================================================
# TAB 3: Documentation
# =========================================================
def documentation():
    st.markdown("## Documentation / Aide")
    card_open("Observance App")
    st.markdown("""
    Cette application permet :
    - Visualisation des rapports environnementaux sous forme de tableau de bord
    - Recherche et analyse
    - Analyse critique des projets
    """)
    st.markdown("**Autrice:** Linh ĐINH  \n**Version:** 1.0")
    card_close()

# =========================================================
# MAIN APP (Sidebar navigation with option_menu)
# =========================================================
st.set_page_config(page_title="Observance", layout="wide", initial_sidebar_state="expanded")
inject_css()

# --- Setup data (download DB + Chroma from Seafile) ---
conn, client = setup_data()

with st.sidebar:
    st.markdown("### OBSERVANCE")
    choice = option_menu(
        menu_title=None,
        options=["Tableau de bord", "Recherche & Analyse", "Documentation / Aide"],
        icons=["bar-chart", "search", "book"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"background-color": PRIMARY, "padding": "0.5rem 0.4rem"},
            "icon": {"color": "#e6fffa", "font-size": "18px"},
            "nav-link": {
                "font-weight": "600",
                "color": "#e6fffa",
                "border-radius": "12px",
                "padding": "8px 12px",
                "margin": "4px 6px",
            },
            "nav-link-selected": {"background-color": PRIMARY_600, "color": "#fff"},
        },
    )

# Route content
if choice == "Tableau de bord":
    dashboard()
elif choice == "Recherche & Analyse":
    recherche_analyse()
else:
    documentation()

