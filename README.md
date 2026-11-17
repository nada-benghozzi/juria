# JurIA : assistant juridique intelligent pour startups et PME tunisiennes


JurIA répond aux questions juridiques à partir d'une base documentaire
(texte, tableaux, figures) grâce à une architecture RAG multimodale,
et cite ses sources.

## Structure

| Fichier | Rôle |
|---|---|
| `ingest.py` | Lit le PDF et construit l'index (à lancer une fois, sur Colab) |
| `rag.py` | Charge l'index et répond aux questions (`ask()`) |
| `app.py` | Interface de chat Streamlit |
| `notebooks/` | Notebook Colab pour lancer l'ingestion |

## Installation

À compléter.

## Utilisation

À compléter.

