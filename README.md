# OBSERVANCE  
**Plateforme d’analyse et de visualisation des avis de l’Autorité environnementale (Ae)**  

<p align="center">
  <img src="images/Observance_logo.png" alt="Logo OBSERVANCE" width="250"/>
</p>

> Outil numérique conçu pour étudier la manière dont les avis de l'**Autorité environnementale (Ae)** sont intégrés dans le processus décisionnel, notamment à travers les **mémoires en réponse** des maîtres d’ouvrage.

## Stack technique

| Domaine | Outils / librairies |
|----------|---------------------|
| Extraction & OCR | `pdfplumber`, `pytesseract`, `Regex` |
| NLP & Vectorisation | `LangChain`, `HuggingFace`, `ChromaDB`, `FTS5` |
| LLM | `Ollama` |
| Frontend | `Streamlit`, `Plotly` |
| Base de données | `SQLite` |
| Déploiement | `Docker`, `Google Cloud VM` |

## Fonctionnalités principales

- **Tableau de bord interactif** : indicateurs, cartes, statistiques.  
- **Recherche plein texte et sémantique** dans les avis et réponses.  
- **Analyse avancée via RAG** : analyse documentaire assistée par l'IA.  
- **Exploration détaillée des projets** et métadonnées associées.

## Captures d’écran

### Page d’accueil
<p align="center">
  <img src="images/page_daccueil.png" alt="Page d'accueil" width="480"/>
</p>

### Tableau de bord
<p align="center">
  <img src="images/page1.png" alt="Dashboard" width="480"/>
</p>

### Carte et projets
<p align="center">
  <img src="images/page2.png" alt="Carte et projets" width="480"/>
</p>

### Recherche et Analyse
<p align="center">
  <img src="images/page3.png" alt="Recherche et Analyse" width="480"/>
</p>

### Analyse avec RAG
<p align="center">
  <img src="images/page4.png" alt="RAG" width="600"/>
</p>


## Application accessible sur 
<p align="center">
  <a href="http://34.38.26.53:8501" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit"/>
  </a>
</p>

## Licence  
MIT — libre de réutilisation avec attribution.

---

## English summary  
**OBSERVANCE** is a data-driven platform using NLP and vector databases to analyze environmental authority opinions (Ae) in France. 
It provides interactive dashboards, semantic search, and RAG-powered AI analysis to support environmental governance research and decision-making.
