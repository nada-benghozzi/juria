# JurIA : assistant d'information juridique pour startups et PME tunisiennes

Projet **Workshop d'Ingénierie**, Faculté des Sciences de Tunis, année universitaire 2026/2027.
Encadrante : **Dr. Asma Amdouni**.

JurIA est un assistant conversationnel qui répond aux questions juridiques d'un dirigeant non juriste, **à partir d'une base documentaire** (texte, **tableaux** et **figures**), et **cite ses sources**. Il repose sur une architecture **RAG multimodale** : au lieu de répondre de mémoire, l'IA retrouve d'abord les passages pertinents, puis rédige une réponse claire en français en renvoyant à ces passages.

> JurIA fournit de l'information juridique, pas un conseil. Pour un enjeu important, consultez un avocat.

## Problème visé

Les startups et PME évoluent dans un environnement juridique complexe, sans service juridique interne et avec peu de moyens pour consulter régulièrement un avocat. JurIA vise un accès **rapide, simple, peu coûteux et traçable** à l'information juridique, dans le contexte tunisien.

## Fonctionnalités

| Page | Description |
|------|-------------|
| **Assistant juridique** | Chat en langage courant, réponses citant leurs sources `[1]`, `[2]`, avec tableaux et figures affichés à côté de la source dont ils proviennent |
| **Analyser un document** | Dépôt d'un PDF (texte) ou d'une photo, confronté à la base juridique |
| **Parcours création** | Checklist des étapes de création d'une société, avec l'aide de l'IA à chaque étape |
| **Tableau de bord** | Usage anonymisé : KPI, thématiques, activité, questions sans réponse |
| **Offres et feuille de route** | Modèle d'abonnement SaaS, évolutions prévues, responsabilité et risques |

Autres points : thème clair / sombre, menu réductible, relances contextuelles, retour utilisateur (pouces), export de la conversation en Markdown, export CSV des tableaux, repli sur la page d'origine du PDF quand un tableau est mal reconnu par l'OCR.

## Architecture

```
                 INGESTION (une seule fois, sur Colab)
PDF ──► extraction (texte / tableaux HTML / images)
    ──► résumés en français (Groq pour texte et tableaux, Gemini pour images)
    ──► embeddings multilingues ──► index FAISS (des résumés)
                                └─► originals.json (contenus originaux)

                 RÉPONSE (à chaque question)
question ──► recherche dans l'index ──► récupération des ORIGINAUX
         ──► Gemini (sources numérotées + figures) ──► réponse citée
         ──► Streamlit (réponse, sources, tableaux, figures)
```

Principe multi-vecteurs : on **indexe des résumés** (plus faciles à retrouver) et on **renvoie les originaux** (texte, tableau HTML, image) au modèle de génération.

### Modèles utilisés

| Rôle | Modèle |
|------|--------|
| Résumés texte et tableaux | `openai/gpt-oss-20b` (Groq) |
| Description des images | `gemini-3.1-flash-lite` (Gemini) |
| Génération de la réponse | `gemini-3.1-flash-lite` (modifiable via `JURIA_GEN_MODEL`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

## Structure du dépôt

```
juria/
├── app.py                    # Interface Streamlit
├── rag.py                    # Backend : load_index(), ask(), ask_document()
├── ingest.py                 # Construction de l'index à partir du PDF
├── assets/                   # Logo.png, Icon.png
├── .streamlit/config.toml    # Configuration Streamlit
├── notebooks/                # Notebook Colab (ingestion + test)
├── data/                     # PDF source (non versionné)
├── index/                    # Index généré (non versionné)
├── logs/                     # Journal d'usage anonymisé (non versionné)
├── .env.example              # Modèle de configuration des clés API
├── requirements.txt          # Dépendances de l'application
└── requirements-ingest.txt   # Dépendances supplémentaires pour l'ingestion
```

## Base documentaire

Document indexé : *« Mon projet d'entreprise, pas à pas : Guide du jeune entrepreneur »* (ONUDI, projet Mashrou3i, 2014), 69 pages en français avec tableaux et schémas.
Résultat de l'ingestion : **70 passages de texte, 46 tableaux, 43 figures**, soit 149 éléments indexés.

> Le guide date de 2014 : certains montants, taux ou procédures ont pu évoluer. Vérifiez-les auprès des organismes concernés.

## Installation

Prérequis : Python 3.10+.

```bash
git clone https://github.com/nada-benghozzi/juria.git
cd juria
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### Clés API

```bash
cp .env.example .env
```

Renseignez dans `.env` :

```
GOOGLE_API_KEY=...   # génération des réponses et description des images
GROQ_API_KEY=...     # ingestion uniquement (résumés)
```

Ne commitez jamais `.env`.

## Utilisation

### 1. Construire l'index (une seule fois, de préférence sur Google Colab)

L'extraction `hi_res` est lourde en mémoire. Utilisez le notebook `notebooks/juria_ingestion.ipynb`, ou en ligne de commande :

```bash
pip install -r requirements-ingest.txt
python ingest.py --pdf data/source.pdf --out index/
# Si les résumés échouent (quota API), sans refaire l'extraction :
python ingest.py --pdf data/source.pdf --out index/ --reuse-extraction
```

Fichiers produits dans `index/` : `index.faiss`, `index.pkl`, `originals.json`, `config.json`, `extraction.json`.
Sur Colab, copiez ce dossier depuis Drive vers `index/` du projet.

### 2. Tester le backend en ligne de commande

```bash
python rag.py "Quelles sont les étapes pour créer une société en Tunisie ?" --index index/
```

### 3. Lancer l'interface

```bash
streamlit run app.py
```

Variables optionnelles : `JURIA_INDEX_DIR` (dossier de l'index), `JURIA_LOG_PATH` (journal d'usage), `JURIA_PDF_PATH` (PDF du guide), `JURIA_GEN_MODEL` (modèle de génération).
Pour afficher la page d'origine du PDF quand un tableau est illisible : placer le PDF à côté de `app.py` (ou `data/`) sous le nom `juria-source.pdf` et installer `pymupdf`.

## Fiabilité et limites

- Réponses construites **uniquement** à partir des sources retrouvées ; si rien de pertinent n'est trouvé, JurIA répond qu'il ne sait pas (`NOT_FOUND`) au lieu d'inventer.
- Les sources sont traitées comme des données, jamais comme des instructions (limitation de l'injection de prompt).
- Recommandation de consulter un avocat quand l'enjeu est élevé.
- Limites connues : qualité de l'OCR sur les pages scannées (tableaux parfois dégradés), quelques résumés de passages non générés lors de l'ingestion (quota API), base limitée à un seul guide daté de 2014, prototype non évalué sur un jeu de questions complet.

| Risque | Mesure de prévention |
|--------|----------------------|
| Hallucinations de l'IA | RAG + sources vérifiables |
| Textes obsolètes | Mise à jour du corpus |
| Fuite de données | Chiffrement et contrôle d'accès |
| Mauvaise interprétation juridique | Orientation vers un avocat |
| Biais de l'IA | Tests et contrôle des réponses |

## Modèle économique (indicatif)

Abonnement SaaS : **Free** (0 DT), **Starter** (49 DT/mois), **Business** (99 DT/mois), **Premium** (199 DT/mois).
Besoin de financement initial estimé : 35 000 à 55 000 DT.

## Feuille de route

1. Enrichissement juridique (fiscal, social, commercial, propriété intellectuelle)
2. Intelligence documentaire (analyse de contrats, détection des clauses à risque)
3. Accessibilité multilingue (arabe et français)
4. Veille juridique intelligente (alertes sur les changements législatifs)
5. Expansion (partenariats avec des professionnels du droit, autres marchés)

Technologies : Streamlit, LangChain, FAISS, Sentence-Transformers, Unstructured, Groq, Google Gemini.