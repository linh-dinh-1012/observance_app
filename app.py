import streamlit as st
import time
import sqlite3
import pandas as pd
import os, requests, zipfile 
import base64
from pathlib import Path
from chromadb import PersistentClient
import plotly.express as px
from shapely import wkt
from shapely.geometry import mapping
from pathlib import Path
from streamlit_option_menu import option_menu 
from streamlit_tags import st_tags 
import streamlit.components.v1 as components
from scripts.rag import answer_query  # Import RAG function
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

MAX_QUERIES = 3
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Clé OpenAI manquante. Configurez OPENAI_API_KEY dans Streamlit Secrets.")
    st.stop()

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

    /* ===== KPI STATUS CARDS ===== */
    .kpi-status{{
        background:#fff;
        border-radius:16px;
        box-shadow:0 4px 18px rgba(0,0,0,.06);
        padding:18px 22px;
        text-align:center;
    }}

    /* SUCCESS */
    .kpi-status.success{{
        border:1px solid #bbf7d0;
    }}
    .kpi-status.success .value{{ color:#16a34a; }}
    .kpi-status.success .icon svg{{ stroke:#16a34a; }}

    /* WARNING */
    .kpi-status.warning{{
        border:1px solid #fde68a;
    }}
    .kpi-status.warning .value{{ color:#d97706; }}
    .kpi-status.warning .icon svg{{ stroke:#d97706; }}

    /* DANGER */
    .kpi-status.danger{{
        border:1px solid #fecaca;
    }}
    .kpi-status.danger .value{{color:#dc2626; }}
    .kpi-status.danger .icon svg{{ stroke:#dc2626; }}

    /* shared */
    .kpi-status .value{{
        font-size:2rem;
        font-weight:800;
        line-height:1;
    }}
    .kpi-status .label{{
        display:flex;
        gap:8px;
        justify-content:center;
        align-items:center;
        margin-top:8px;
        color:#6b7280;
        font-weight:600;
    }}
    .kpi-status .icon{{
        width:18px;
        height:18px;
        display:inline-flex;
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
    if isinstance(value, (int, float)):
        display_value = f"{value:,}"
    else:
        display_value = str(value)

    st.markdown(
        f"""
        <div class="kpi">
            <div class="value">{display_value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def kpi_card_auto(label: str, value_pct: float):
    """
    value_pct: valeur entre 0 et 100
    """
    if value_pct >= 80:
        status = "success"
    elif value_pct >= 50:
        status = "warning"
    else:
        status = "danger"

    display_value = f"{value_pct:.0f}%"

    check_svg = """
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M9 12.5l2 2 4-5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M12 22c5.52 0 10-4.48 10-10S17.52 2 12 2 2 6.48 2 12s4.48 10 10 10Z"
            stroke-width="2.2"/>
    </svg>
    """

    st.markdown(
        f"""
        <div class="kpi-status {status}">
            <div class="value">{display_value}</div>
            <div class="label">
                <span class="icon">{check_svg}</span>
                <span>{label}</span>
            </div>
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
# DATABASE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "observance.db"   
CHROMA_PATH = BASE_DIR / "chroma_store"  



def run_query(query, params=()):
    """Execute a read-only SQL query and return a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


# =========================================================
# TAB 1: Page d'accueil
# =========================================================
def home():
    base_dir = Path(__file__).resolve().parent
    logo_path = base_dir / "assets" / "logo_observance.png"

    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()
        logo_src = f"data:image/png;base64,{logo_base64}"
    else:
        logo_src = ""

    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        .block-container {padding: 0; margin: 0; max-width: 100%;}
        html, body {height: 100%; margin: 0;}
        </style>
    """, unsafe_allow_html=True)

    # CSS hero section
    st.markdown(f"""
    <style>
    @keyframes slideInLeft {{
        0% {{opacity: 0; transform: translateX(-50px);}}
        100% {{opacity: 1; transform: translateX(0);}}
    }}
    @keyframes slideInRight {{
        0% {{opacity: 0; transform: translateX(50px);}}
        100% {{opacity: 1; transform: translateX(0);}}
    }}

    .hero {{
        display: flex;
        align-items: center;
        justify-content: center;
        height: 70vh;
        background: #ffffff;
        padding: 0 6%;
        box-sizing: border-box;
    }}
    .hero-left {{
        flex: 1;
        text-align: center;
        animation: slideInLeft 1s ease-out;
    }}
    .hero-left img {{
        width: 420px;
        max-width: 95%;
    }}
    .hero-right {{
        flex: 1;
        text-align: left;
        padding-left: 3rem;
        animation: slideInRight 1s ease-out;
    }}
    .hero-right h1 {{
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f4336;
        margin-bottom: 1.5rem;
        line-height: 1.3;
    }}
    .hero-right p {{
        font-size: 1.1rem;
        color: #374151;
        margin-bottom: 2.5rem;
        max-width: 520px;
        line-height: 1.6;
    }}
    .stButton>button {{
        background:#047857;
        color:#fff !important;
        font-weight:600;
        padding:0.9rem 1.8rem;
        border-radius:999px;
        border:none;
        transition:background 0.3s,transform 0.2s;
        font-size:1.05rem;
        cursor:pointer;
    }}
    .stButton>button:hover {{
        background:#065f46;
        transform:translateY(-3px);
    }}
    .btn-wrapper {{
        text-align: left;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Hero HTML
    st.markdown(f"""
    <div class="hero">
        <div class="hero-left">
            <img src="{logo_src}" alt="Logo Observance">
        </div>
        <div class="hero-right">
            <h1>Plateforme d’analyse et de visualisation<br>des avis de l’Autorité environnementale (Ae)</h1>
            <p>Un outil numérique conçu pour explorer, comprendre et suivre la trajectoire de prise en compte des enjeux environnementaux dans les processus décisionnels.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # "Découvrir" button
    _, col_right = st.columns([2, 1], gap="large")  
    with col_right:
        st.markdown('<div style="margin-top:-1.5rem;">', unsafe_allow_html=True) 
        if st.button("Découvrir", key="discover-btn"):
            st.session_state["page"] = "presentation"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 2: Dashboard
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
    taux_reponse = round((reponse_count/avis_count) * 100, 1)

    # KPI row
    k1, k2, k3 = st.columns([1,1,1])
    with k1: kpi_card("Avis", avis_count)
    with k2: kpi_card("Réponses", reponse_count)
    with k3: kpi_card_auto(
        "Projets avec réponse",
        taux_reponse
    )
    card_close()

    # --- Evolution of avis over time (line chart) ---
    left, right = st.columns([1.35, 1])
    with left:
        card_open("Évolution du nombre des avis au fil du temps")
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
                fig_year.update_traces(
                    hovertemplate="Année: %{x}<br>Nombre d'avis: %{y}<extra></extra>"
                )

                fig_year.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    hoverlabel=dict(font_size=14, font_family="Arial"),
                    xaxis_title="Année",
                    yaxis_title="Nombre d'avis",
                )

                fig_year.update_traces(text=df_year["count"], textposition="top center")
                fig_year.update_xaxes(type="category", showgrid=False)
                fig_year.update_yaxes(showgrid=False)

                st.plotly_chart(fig_year, use_container_width=True, key="evolution_avis")
        card_close()

    # --- Pie chart: Distribution of critique levels ---
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
            # --- Clean & map critique levels ---
            df_crit["avis_code"] = (
                pd.to_numeric(df_crit["avis_critique"], errors="coerce")
                .round()
                .astype("Int64")
            )

            labels_map = {
                1: "Peu critique",
                2: "Mitigé",
                3: "Assez critique",
                4: "Très critique",
            }

            df_crit["avis_label"] = df_crit["avis_code"].map(labels_map)

            df_plot = (
                df_crit
                .dropna(subset=["avis_label"])
                .groupby("avis_label", as_index=False)["count"]
                .sum()
            )

            # Ordre métier clair
            order = [
                "Peu critique",
                "Mitigé",
                "Assez critique",
                "Très critique"
            ]

            df_plot["avis_label"] = pd.Categorical(
                df_plot["avis_label"],
                categories=order,
                ordered=True
            )
            df_plot = df_plot.sort_values("avis_label")

            blues = px.colors.sequential.Blues
            blue_scale = [
                blues[1],  # Peu critique  (très clair)
                blues[3],  # Mitigé
                blues[5],  # Assez critique
                blues[8],  # Très critique (foncé)
            ]

            color_map = {
                "Peu critique": blue_scale[0],
                "Mitigé": blue_scale[1],
                "Assez critique": blue_scale[2],
                "Très critique": blue_scale[3],
            }
            
            fig_pie = px.pie(
                df_plot,
                names="avis_label",
                values="count",
                hole=0.6,
                color="avis_label",
                color_discrete_map=color_map,
                category_orders={"avis_label": order},   
            )

            fig_pie.update_traces(
                texttemplate="%{percent:.2%}",
                textposition="outside",
                textfont=dict(size=13),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:,} avis<br>"
                    "<extra></extra>"
                ),
                sort=False
            )

            fig_pie.update_layout(
                legend=dict(
                    title="Niveau critique",
                    x=1.02,
                    y=0.5,
                    font=dict(size=13),
                    traceorder="normal"                  
                ),
                margin=dict(l=10, r=120, t=10, b=10),
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        card_close()

    # --- Top 5 critical projects ---
    card_open("Top 5 des projets critiqués")
    df_top = run_query("""
    WITH ranked AS (
        SELECT
            titre,
            nb_recommandations,
            avis_critique,
            ROW_NUMBER() OVER (
                PARTITION BY titre
                ORDER BY 
                    CAST(avis_critique AS INTEGER) DESC,
                    nb_recommandations DESC
            ) AS rn
        FROM projects
    )
        SELECT 
            titre,
            nb_recommandations,
            avis_critique
        FROM ranked
        WHERE rn = 1
        ORDER BY 
            CAST(avis_critique AS INTEGER) DESC,
            nb_recommandations DESC
        LIMIT 5;
    """)

    # Reset index to start at 1
    df_top.index = df_top.index + 1
    df_top.index.name = "#"

    # Rename columns (with bold labels)
    df_top = df_top.rename(columns={
        "titre": "Titre",
        "nb_recommandations": "Nombre de recommandations",
        "avis_critique": "Niveau critique de l'avis"
    })

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
        fig_map.update_layout(
            coloraxis_colorbar=dict(
                title=dict(
                    text="Nombre de recommandations",  
                    font=dict(size=14)
                ),
                tickfont=dict(size=12)
            ),
        )

        fig_map.update_traces(marker_line_width=2, marker_line_color="black")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, hoverlabel=dict(font_size=14, font_family="Arial"))
        fig_map.update_geos(fitbounds="locations")
        st.plotly_chart(fig_map, use_container_width=True, key="map_projects")
    map_projects()
    card_close()

# =========================================================
# TAB 4: Search & Analyse
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
        ORDER BY CAST({col} AS INTEGER)
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
    st.markdown("<p style='color:#4b5563;'>Cette section vous permet de choisir un projet, consulter ses informations clés et poser vos questions à l’IA pour une analyse contextualisée.</p>", unsafe_allow_html=True)
    st.markdown("""
            <p style='color:#92400e; background:#fff7ed; padding:12px 14px; border-radius:10px; border:1px solid #fed7aa;'>
            <strong>Version démo</strong><br>
            Cette application est proposée à des fins de démonstration.
            Elle ne contient qu'un nombre limité documents par rapport à la version officielle.<br>
            Pour tester la fonctionnalité <strong>Recherche augmentée par IA (RAG)</strong>, 
            nous vous invitons de sélectionner <strong>le premier projet de la liste</strong>,
            puis de poser votre question dans la zone d'analyse en bas de page affichée après sélection. 
            </p>
            """, unsafe_allow_html=True)

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
            # Remap + trier par ordre numérique
            labels_map = {
                "1": "1. Peu critique",
                "2": "2. Mitigé",
                "3": "3. Assez critique à critique",
                "4": "4. Très critique"
            }
            critiques_raw_sorted = sorted([c for c in critiques_raw if c is not None], key=lambda x: float(x))
            critiques_labels = [labels_map.get(str(c), str(c)) for c in critiques_raw_sorted]
            critique_sel = st.selectbox(
                "Niveau critique de l'avis",
                [""] + critiques_labels,
                index=0
            )
        keywords = st_tags(
            label="Mots-clés",
            text="Entrez un mot-clé",
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
    if st.session_state.query_count >= MAX_QUERIES:
        st.warning("⚠️ Version démo : limite de requêtes atteinte pour cette session.")
        st.stop()

    st.session_state.query_count += 1

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
            with st.spinner("Traitement en cours… cela peut prendre quelques instants"):
                try:
                    answer = answer_query(q, project_id=selected_id)

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
# TAB 5: À propos
# =========================================================
def a_propos():
    st.markdown("## ℹ️ À propos")
    st.markdown(""" **OBSERVANCE** est une plateforme d’analyse de données environnementales visant à comprendre comment les recommandations de l'**Autorité environnementale (Ae)** sont prises en compte par les maîtres d'ouvrage dans leurs projets. 
                                                                       
    Ce travail s'inscrit dans la continuité du projet **PEGASE** (2018-2023), financé par le programme **ITTECOP** et soutenu par le **ministère de la Transition écologique** et de l’**ADEME**. PEGASE analysait la gouvernance de l'évaluation environnementale en France à la suite de la réforme de 2016, notamment le rôle et l’influence de l’**Autorité environnementale (Ae)** et de ses **missions régionales (MRAe)**.                  

    **Direction scientifique du projet PEGASE :** **Cécile Blatrix** & **Nathalie Frascaria-Lacoste**   

    Pour en savoir plus sur ce projet :                            
                                                                       
    👉 [Rapport PEGASE – ITTECOP](https://ittecop.fr/fr/ressources/telechargements/rapport-final/rapport-ittecop-apr-2017/1391-apr-2017-pegase-rf/file) 
                
    👉 [Projet OBSERVANCE](https://www.canva.com/design/DAGyXlGTcrQ/U1Fswd8NBQJTfmo6HWsqPw/view) """) 
    
    st.markdown("---")
    st.markdown("""
    **👩‍💻 Développeuse :** Linh Đinh            
    **📧 Email :** [contact@linhdinh.fr](mailto:contact@linhdinh.fr)  
    **🔗 LinkedIn :** [Linh Đinh](https://www.linkedin.com/in/thi-thuy-linh-dinh/)
    """)

# =========================================================
# MAIN APP (Sidebar navigation with option_menu)
# =========================================================
st.set_page_config(page_title="Observance",
                   page_icon="assets/flavicon.png", 
                   layout="wide", 
                   initial_sidebar_state="expanded")
inject_css()


# Route content
if "page" not in st.session_state:
    st.session_state["page"] = "home"

current_page = st.session_state["page"]

if current_page == "home":
    home()

else:
    with st.sidebar:
        st.markdown("### OBSERVANCE")
        choice = option_menu(
            menu_title=None,
            options=[
                "Tableau de bord",
                "Recherche & Analyse",
                "À propos"
            ],
            icons=["bar-chart", "search", "info-circle"],
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

    # --- Routing logic
    if choice == "Tableau de bord":
        dashboard()
    elif choice == "Recherche & Analyse":
        recherche_analyse()
    elif choice == "À propos":
        a_propos()