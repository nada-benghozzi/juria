"""
app.py : interface Streamlit de JurIA (assistant juridique pour startups et PME tunisiennes).

Pages :
    Assistant juridique       chat avec réponses sourcées ; chaque source affiche son tableau ou sa figure
    Analyser un document      PDF (texte) ou photo confronté à la base juridique
    Parcours création         checklist de création de société, avec aide de l'IA à chaque étape
    Tableau de bord           usage anonymisé : KPI, thématiques, activité, questions sans réponse
    Offres et feuille de route   modèle d'abonnement, évolutions, responsabilité

Base indexée : « Mon projet d'entreprise, pas à pas : Guide du jeune entrepreneur »
(ONUDI / projet Mashrou3i, 2014). Les thématiques et les questions d'exemple en reprennent le plan.

Contrat avec le backend (rag.py) :
    rag.load_index(path)                       -> configuration de l'index
    rag.ask(question)                          -> dict (answer, answerable, sources, images, tables)
    rag.ask_document(question, data, mime)     -> même format

    Chaque source : {"type": "text"|"table"|"image", "pages": [..], "score": float,
                     "cited": bool, "extrait": str}
    « cited »  = la réponse renvoie à ce passage par [n] ; sinon le passage a seulement été
    retrouvé dans l'index et fourni au modèle comme contexte (« consultée »).

Ressources graphiques attendues : assets/logo.png et assets/icon.png
Optionnel : le PDF du guide (juria-source.pdf, ou JURIA_PDF_PATH) + `pip install pymupdf` pour afficher
la page d'origine quand un tableau extrait est illisible.

Lancement :
    streamlit run app.py
"""

import base64
import csv
import html
import inspect
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime
from functools import lru_cache
from html.parser import HTMLParser

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
INDEX_DIR = os.environ.get("JURIA_INDEX_DIR", "index/")
LOG_PATH = os.environ.get("JURIA_LOG_PATH", "logs/usage.jsonl")
MAX_QUESTION_CHARS = 1000
MAX_UPLOAD_MB = 10
SEUIL_QUALITE_TABLEAU = 0.78   # en dessous, la lecture OCR est jugée dégradée
MAX_MOTS_SUIVI = 6             # une question plus courte est traitée comme une relance
JOURS_ACTIVITE = 14            # nombre de jours affichés dans le graphique d'activité

# (clé, libellé, icône Material)
NAV = [
    ("assistant", "Assistant juridique", "chat"),
    ("document", "Analyser un document", "description"),
    ("parcours", "Parcours création", "checklist"),
    ("dashboard", "Tableau de bord", "monitoring"),
    ("about", "Offres et feuille de route", "workspace_premium"),
]

# Thématiques alignées sur le plan du guide indexé (chapitres I à IV et annexes).
# « id » sert à la couleur ; « keywords » au suivi d'usage (détection par mots-clés).
# Retirez les questions qui renvoient « information introuvable » pendant vos tests.
TOPICS = {
    "Création de société": {
        "id": "creation", "icon": "building",
        "desc": "Formalités, registre du commerce, JORT",
        "keywords": ["créer", "création", "formalité", "registre du commerce", "jort", "greffe",
                     "immatricul", "déclaration d'investissement", "dépôt", "formalité unique",
                     "statuts"],
        "questions": ["Quelles sont les étapes pour créer une société en Tunisie ?",
                      "Quels documents faut-il pour le dépôt définitif au registre du commerce ?"],
    },
    "Forme juridique": {
        "id": "forme", "icon": "scale",
        "desc": "SARL, SUARL, SA, capital, associés",
        "keywords": ["sarl", "suarl", "société anonyme", "forme juridique", "gérant", "capital",
                     "associé", "commandite", "sca", "responsabilité limitée"],
        "questions": ["Quelle est la différence entre une SARL et une SA ?",
                      "Quel est le capital minimum et le nombre d'associés pour une SA ?"],
    },
    "Fiscalité et social": {
        "id": "fisc", "icon": "percent",
        "desc": "IRPP, IS, TVA, identifiant fiscal, cotisations sociales",
        "keywords": ["impôt", "irpp", "impôt sur les sociétés", "fiscal", "tva", "taxe",
                     "identifiant fiscal", "droit d'enregistrement", "cotisation", "cnss",
                     "sécurité sociale", "affiliation", "patronal", "salaire", "recrut", "employé"],
        "questions": ["Comment obtenir un identifiant fiscal ?",
                      "Quels sont les taux de l'IRPP selon le revenu imposable ?",
                      "Comment l'État prend-il en charge la cotisation patronale ?"],
    },
    "Financement": {
        "id": "fin", "icon": "banknote",
        "desc": "FONAPRAM, FOPRODI, SICAR, BTS, BFPME, SOTUGAR",
        "keywords": ["financ", "fonapram", "foprodi", "sicar", "fcpr", "bts", "bfpme", "sotugar",
                     "amorçage", "microcrédit", "garantie", "capital risque", "ligne de crédit",
                     "fonds propres", "crédit"],
        "questions": ["Comment fonctionne le FOPRODI pour les nouveaux promoteurs ?",
                      "Qu'est-ce que le FONAPRAM et quelles primes accorde-t-il ?",
                      "Quel est le rôle de la SOTUGAR ?"],
    },
    "Incitations": {
        "id": "incit", "icon": "gift",
        "desc": "Avantages fiscaux, primes, régimes spécifiques",
        "keywords": ["incitation", "avantage", "prime", "exonération", "déduction", "riictic",
                     "régime d'incitation", "ontt", "dégrèvement"],
        "questions": ["Quelles sont les incitations fiscales à la création d'un projet ?",
                      "Quelles incitations financières existent pour un nouveau projet ?"],
    },
    "Accompagnement": {
        "id": "accomp", "icon": "compass",
        "desc": "Pépinières, ANETI, essaimage, congé de création",
        "keywords": ["pépinière", "centre d'affaires", "aneti", "essaimage", "essaimé", "congé",
                     "accompagn", "appui", "promoteur", "plan d'affaires", "idée de projet"],
        "questions": ["Que proposent les pépinières d'entreprises ?",
                      "Qu'est-ce que l'essaimage et quels en sont les avantages ?",
                      "Comment bénéficier du congé pour création d'entreprise ?"],
    },
}

# Couleur (texte, fond) de chaque thématique, déclinée en clair / sombre.
# Toutes restent dans la gamme bleu-turquoise du thème, avec des accents distincts.
TOPIC_COLORS = {
    "light": {
        "creation": ("#0B5E8E", "#E3F1F8"), "forme": ("#4A56B5", "#E8EAF8"),
        "fisc": ("#0E8A7D", "#DDF3EF"), "fin": ("#2E8B4F", "#E0F3E6"),
        "incit": ("#B7791F", "#FBEFD0"), "accomp": ("#7B4FB0", "#EFE6F8"),
        "autre": ("#5A7084", "#EAF1F7"),
    },
    "dark": {
        "creation": ("#4DB6E6", "#15314B"), "forme": ("#9AA5F0", "#1E2347"),
        "fisc": ("#4FD1BE", "#123A36"), "fin": ("#7FD9A4", "#143A28"),
        "incit": ("#F0C873", "#43361A"), "accomp": ("#C3A0EE", "#2E2147"),
        "autre": ("#93A7BC", "#172740"),
    },
}


def topic_id(nom):
    return TOPICS.get(nom, {}).get("id", "autre")


def topic_style(nom):
    """Variables CSS (--c texte, --b fond) de la thématique, à placer dans un attribut style."""
    i = topic_id(nom)
    return f"--c:var(--tc-{i});--b:var(--tb-{i})"


# Parcours de création de société (étapes issues des tableaux 2.3 et 2.4 du guide indexé).
CHECKLIST = [
    ("Dépôt provisoire des statuts", "Greffe du tribunal",
     "Quels documents faut-il pour le dépôt provisoire des statuts ?"),
    ("Enregistrement des statuts", "Recette des finances",
     "Comment enregistrer les statuts à la recette des finances ?"),
    ("Dépôt définitif au registre du commerce", "Greffe du tribunal",
     "Quels documents faut-il pour le dépôt définitif au registre du commerce ?"),
    ("Publication au JORT", "Journal Officiel (JORT)",
     "Quelles sont les étapes pour publier au JORT ?"),
    ("Immatriculation au registre du commerce", "Greffe du tribunal",
     "Comment se déroule l'immatriculation au registre du commerce ?"),
]

DOC_PROMPTS = ["Résume ce document", "Quelles clauses sont risquées ?",
               "Ce document est-il complet ?", "Quelles obligations en découlent ?"]

# Compatibilité entre versions de Streamlit
_BTN = inspect.signature(st.button).parameters
STRETCH = {"width": "stretch"} if "width" in _BTN else {"use_container_width": True}


def ic(nom):
    """Argument d'icône Material pour un bouton (ignoré si la version ne le gère pas)."""
    return {"icon": f":material/{nom}:"} if "icon" in _BTN else {}


# Type de source : (icône, libellé, couleur CSS)
TYPES_FR = {
    "text": ("file-text", "Texte", "var(--primary)"),
    "table": ("table", "Tableau", "var(--tc-fisc)"),
    "image": ("image", "Figure", "var(--tc-accomp)"),
}

# Icônes vectorielles (style Lucide) pour les éléments HTML
_SVG = {
    "library": '<path d="m16 6 4 14"/><path d="M12 6v14"/><path d="M8 8v12"/><path d="M4 4v16"/>',
    "table": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M12 3v18"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.9.7-1.9 1-3 1s-2.1-.3-3-1Z"/><path d="m2 16 3-8 3 8c-.9.7-1.9 1-3 1s-2.1-.3-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "alert": '<path d="m21.7 18-8-14a2 2 0 0 0-3.5 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.7 9a1 1 0 0 1-.6 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.2-2.7a1.2 1.2 0 0 1 1.6 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "map": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "building": '<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M12 6h.01"/><path d="M16 6h.01"/><path d="M8 10h.01"/><path d="M12 10h.01"/><path d="M16 10h.01"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/>',
    "percent": '<line x1="19" x2="5" y1="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "banknote": '<rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/>',
    "gift": '<rect x="3" y="8" width="18" height="4" rx="1"/><path d="M12 8v13"/><path d="M19 12v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-7"/><path d="M7.5 8a2.5 2.5 0 0 1 0-5A4.8 8 0 0 1 12 8a4.8 8 0 0 1 4.5-5 2.5 2.5 0 0 1 0 5"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "thumb": '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
    "tag": '<path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4z"/><circle cx="7.5" cy="7.5" r=".5"/>',
}


def svg(nom, taille=15):
    return (f'<svg class="jr-ico" width="{taille}" height="{taille}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f"{_SVG[nom]}</svg>")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def trouver_asset(*noms):
    """Cherche un fichier image (insensible à la casse) dans assets/, à côté de app.py et dans le dossier courant."""
    dossiers = [os.path.join(BASE_DIR, "assets"), BASE_DIR, os.path.join(os.getcwd(), "assets"), os.getcwd()]
    cibles = {n.lower() for n in noms}
    for d in dossiers:
        try:
            for f in os.listdir(d):
                if f.lower() in cibles:
                    return os.path.join(d, f)
        except OSError:
            continue
    return None


def preparer_assets():
    """Retourne (logo complet, emblème). L'emblème est découpé dans le logo s'il n'existe pas."""
    logo = trouver_asset("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")
    icon = trouver_asset("icon.png", "emblem.png", "favicon.png")
    if logo and not icon:
        try:
            from PIL import Image
            im = Image.open(logo).convert("RGB")
            w, h = im.size
            e = im.crop((int(.08 * w), int(.03 * h), int(.93 * w), int(.65 * h)))
            s = max(e.size)
            c = Image.new("RGB", (s, s), (255, 255, 255))
            c.paste(e, ((s - e.width) // 2, (s - e.height) // 2))
            icon = os.path.join(tempfile.gettempdir(), "juria_emblem.png")
            c.resize((256, 256)).save(icon)
        except Exception:
            icon = None
    return logo, icon


LOGO_PATH, ICON_PATH = preparer_assets()


@lru_cache(maxsize=None)
def image_uri(chemin):
    """Image encodée en data URI (pour l'inclure dans du HTML) ; chaîne vide si absente."""
    if not chemin:
        return ""
    try:
        with open(chemin, "rb") as f:
            mime = "image/jpeg" if chemin.lower().endswith((".jpg", ".jpeg")) else "image/png"
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="JurIA · Assistant juridique",
                   page_icon=ICON_PATH,
                   layout="wide", initial_sidebar_state="expanded")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("page", "assistant")
st.session_state.setdefault("theme", "light")
st.session_state.setdefault("mini", False)


# ---------------------------------------------------------------------------
# Thème (clair / sombre) et barre latérale réductible
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "bg": "#F3F7FB", "surface": "#FFFFFF", "surface2": "#EAF1F7", "border": "#D9E3EC",
        "text": "#0F2A3F", "muted": "#5A7084", "primary": "#0B5E8E", "accent": "#1BA5C8",
        "soft": "#E3F1F8", "side_a": "#2588B8", "side_b": "#0B4A7D",
        "hero_a": "#0A3D62", "hero_b": "#1389AE",
        "ok_bg": "#DDF3E7", "ok_fg": "#14683A", "mid_bg": "#FBEFD0", "mid_fg": "#8A5A00",
        "low_bg": "#FBE0E0", "low_fg": "#9B2222", "warn_bg": "#FFF6E0", "warn_bd": "#E0A526",
        "shadow": "0 1px 3px rgba(15,42,63,.07)",
    },
    "dark": {
        "bg": "#0A1220", "surface": "#111C2E", "surface2": "#172740", "border": "#243650",
        "text": "#E6EEF7", "muted": "#93A7BC", "primary": "#4DB6E6", "accent": "#2BC3E3",
        "soft": "#15314B", "side_a": "#1B567A", "side_b": "#0C1A2E",
        "hero_a": "#0F3A5C", "hero_b": "#0E7490",
        "ok_bg": "#143A28", "ok_fg": "#7FD9A4", "mid_bg": "#43361A", "mid_fg": "#F0C873",
        "low_bg": "#472123", "low_fg": "#F0A0A0", "warn_bg": "#2E2814", "warn_bd": "#B8860B",
        "shadow": "0 1px 3px rgba(0,0,0,.4)",
    },
}

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/*VARS*/

/* ---------- Base ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp textarea, .stApp input, .stApp button { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
.block-container { max-width: 980px; padding-top: 2.2rem; padding-bottom: 6rem; }
hr { border-color: var(--border) !important; }

/* ---------- Barre latérale (toujours sombre) ---------- */
section[data-testid="stSidebar"] { width: __W__ !important; min-width: __W__ !important; max-width: __W__ !important;
    transform: none !important; border-right: 0 !important;
    background: linear-gradient(to bottom, var(--side-a), var(--side-b)) !important; }
section[data-testid="stSidebar"] > div { width: __W__ !important; background: transparent !important; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarContent"] { overflow-x: hidden; padding: 0 .5rem; }
[data-testid="stSidebarUserContent"] { padding: 1rem .35rem 1rem .35rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] li { color: #DCE9F5; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: rgba(220,233,245,.6) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14) !important; margin: .6rem 0; }
.stApp [data-testid="stSidebar"] .stButton button, .stApp [data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: transparent !important; border: 0 !important; box-shadow: none !important; border-radius: 10px;
    justify-content: flex-start !important; padding: .5rem .8rem; min-height: 2.4rem; width: 100%; }
.stApp [data-testid="stSidebar"] .stButton button > div {
    display: flex; align-items: center; justify-content: flex-start !important; gap: .65rem; width: 100%; }
.stApp [data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {
    display: inline-flex; align-items: center; justify-content: center; flex: 0 0 1.25rem; width: 1.25rem; }
.stApp [data-testid="stSidebar"] .st-key-btn_mini_top button {
    width: 2.4rem !important; min-height: 2.4rem; padding: .5rem !important; margin-left: auto;
    justify-content: center !important; }
.stApp [data-testid="stSidebar"] .st-key-btn_mini_top button > div { justify-content: center !important; }
.stApp [data-testid="stSidebar"] .st-key-btn_mini_top button [data-testid="stMarkdownContainer"] { display: none !important; }
.stApp [data-testid="stSidebar"] .stButton button p, .stApp [data-testid="stSidebar"] .stButton button span,
.stApp [data-testid="stSidebar"] [data-testid="stDownloadButton"] button p {
    color: #DCE9F5 !important; text-align: left; margin: 0; font-weight: 500; }
.stApp [data-testid="stSidebar"] .stButton button:hover, .stApp [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background: rgba(255,255,255,.09) !important; }
.stApp [data-testid="stSidebar"] .stButton button[kind="primary"],
.stApp [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] { background: rgba(255,255,255,.17) !important; }
.stApp [data-testid="stSidebar"] .stButton button[kind="primary"] p, .stApp [data-testid="stSidebar"] .stButton button[kind="primary"] span,
.stApp [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] p,
.stApp [data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] span { color: #fff !important; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.12) !important; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p, [data-testid="stSidebar"] [data-testid="stExpander"] summary span { color: #DCE9F5 !important; }
/*MINI*/
.jr-logo { border-radius: 12px; padding: .35rem .5rem; width: 128px; margin: .2rem auto 1.5rem auto; }
.jr-logo img { display: block; width: 100%; max-height: 92px; object-fit: contain; margin: 0 auto; }
.jr-logo.mini { width: 46px; padding: .2rem; border-radius: 10px; margin-bottom: 1.2rem; }
.jr-wordmark { font-size: 1.5rem; font-weight: 800; color: #fff !important; padding: .2rem 0 1.5rem .6rem; }
.jr-wordmark span { color: #5CCBE8 !important; }
.jr-side-note { background: rgba(255,255,255,.08); border-radius: 10px; padding: .65rem .8rem; font-size: .76rem;
    line-height: 1.45; color: rgba(220,233,245,.8); margin-top: .6rem; }
.jr-side-title { font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: rgba(220,233,245,.5); margin: .9rem 0 .2rem .4rem; }

/* ---------- Titres et bandeau ---------- */
.jr-title { font-size: 1.55rem; font-weight: 700; color: var(--text); margin: 0 0 .15rem 0; }
.jr-subtitle { color: var(--muted); margin-bottom: 1.1rem; }
.jr-hero { background: linear-gradient(120deg, var(--hero-a), var(--hero-b)); border-radius: 16px;
    padding: 1.6rem 1.9rem; margin-bottom: 1.4rem; box-shadow: 0 8px 24px rgba(10,61,98,.18); }
.jr-hero h1 { font-size: 1.85rem; margin: 0 0 .3rem 0; padding: 0; color: #fff !important; }
.jr-hero p { color: rgba(255,255,255,.92) !important; margin: 0 0 .9rem 0; font-size: 1rem; }
.jr-feature { display: inline-flex; align-items: center; gap: .4rem; background: rgba(255,255,255,.17);
    color: #fff !important; border-radius: 999px; padding: .25rem .8rem; margin: 0 .4rem .35rem 0;
    font-size: .8rem; font-weight: 500; }
.jr-ico { vertical-align: -2px; flex-shrink: 0; }

/* ---------- Composants Streamlit (sélecteurs génériques, sans dépendre de stMain) ---------- */
[data-testid="stChatMessage"] { background: var(--surface) !important; border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem 1.2rem; box-shadow: var(--shadow); }
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
[data-testid="stChatInput"] { background: var(--surface) !important; border: 1px solid var(--border); border-radius: 14px; }
[data-testid="stChatInput"] div, [data-testid="stChatInput"] textarea { background: transparent !important; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
.stApp textarea, .stApp input { color: var(--text) !important; }
[data-baseweb="textarea"], [data-baseweb="base-input"], [data-baseweb="input"] { background: var(--surface) !important;
    border-color: var(--border) !important; }
.stApp textarea, .stApp [data-baseweb="base-input"] input { background: var(--surface) !important; }
.stApp .stButton button, .stApp [data-testid="stDownloadButton"] button {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px; box-shadow: none;
    height: auto; min-height: 2.5rem; padding: .5rem .8rem; white-space: normal; line-height: 1.35; }
.stApp .stButton button p, .stApp [data-testid="stDownloadButton"] button p { color: var(--text); margin: 0; }
.stApp .stButton button:hover, .stApp [data-testid="stDownloadButton"] button:hover { border-color: var(--accent); }
.stApp .stButton button:hover p { color: var(--primary); }
.stApp .stButton button[kind="primary"], .stApp .stButton button[data-testid="stBaseButton-primary"] {
    background: var(--primary); border-color: var(--primary); }
.stApp .stButton button[kind="primary"] p, .stApp .stButton button[data-testid="stBaseButton-primary"] p { color: #fff; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--border) !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] label * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface2) !important; border: 1px solid var(--border) !important; }
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px; }
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: var(--text); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: .8rem 1rem; box-shadow: var(--shadow); }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--text) !important; }
[data-testid="stAlert"] { background: var(--surface2) !important; border: 1px solid var(--border); }
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); }
button[role="tab"] p { color: var(--muted); }
button[role="tab"][aria-selected="true"] p { color: var(--primary); font-weight: 600; }

/* ---------- Thématiques colorées (--c : texte, --b : fond, posés en style inline) ---------- */
.jr-thead { display: flex; align-items: center; gap: .7rem; background: var(--b); border-left: 4px solid var(--c);
    border-radius: 10px; padding: .6rem .8rem; margin-bottom: .45rem; }
.jr-thead .ico { color: var(--c); display: flex; flex-shrink: 0; }
.jr-thead b { display: block; color: var(--text); font-size: .98rem; line-height: 1.25; }
.jr-thead small { display: block; color: var(--muted); font-size: .76rem; line-height: 1.3; margin-top: .1rem; }
.jr-tag { display: inline-flex; align-items: center; gap: .35rem; background: var(--b); color: var(--c);
    border-radius: 999px; padding: .15rem .65rem; margin: 0 .35rem .3rem 0; font-size: .78rem; font-weight: 600; }

/* ---------- Réponses ---------- */
.jr-cite { display: inline-flex; align-items: center; justify-content: center; min-width: 1.25rem; height: 1.25rem;
    padding: 0 .3rem; border-radius: 999px; background: var(--soft); color: var(--primary); font-size: .72rem;
    font-weight: 700; margin: 0 .1rem; vertical-align: text-top; }
.jr-meta { margin: .7rem 0 .4rem 0; }
.jr-chip { display: inline-flex; align-items: center; gap: .35rem; background: var(--surface2); color: var(--muted);
    border-radius: 999px; padding: .15rem .65rem; margin: 0 .35rem .3rem 0; font-size: .78rem; }
.jr-chip.high { background: var(--ok-bg); color: var(--ok-fg); }
.jr-chip.mid { background: var(--mid-bg); color: var(--mid-fg); }
.jr-chip.low { background: var(--low-bg); color: var(--low-fg); }
.jr-callout { display: flex; gap: .6rem; background: var(--warn-bg); border-left: 4px solid var(--warn-bd);
    border-radius: 8px; padding: .7rem 1rem; margin: .8rem 0; font-size: .9rem; color: var(--text); }
.jr-callout svg { margin-top: 2px; color: var(--warn-bd); }
.jr-empty { background: var(--surface2); border: 1px dashed var(--border); border-radius: 12px; padding: 1rem 1.2rem;
    color: var(--text); line-height: 1.55; }
.jr-empty b { display: flex; align-items: center; gap: .4rem; margin-bottom: .3rem; }

/* ---------- Tableaux ---------- */
.jr-table { overflow: auto; max-height: 420px; border: 1px solid var(--border); border-radius: 10px; }
.jr-table table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .85rem; }
.jr-table th { position: sticky; top: 0; background: var(--soft); color: var(--primary); text-align: left;
    padding: .55rem .7rem; border-bottom: 2px solid var(--border); font-weight: 600; }
.jr-table td { padding: .5rem .7rem; border-bottom: 1px solid var(--border); vertical-align: top; min-width: 90px;
    line-height: 1.45; color: var(--text); }
.jr-table tbody tr:nth-child(even) td { background: var(--surface2); }

/* ---------- Sources ---------- */
.jr-source { border: 1px solid var(--border); border-left: 4px solid var(--c, var(--primary)); border-radius: 10px;
    padding: .7rem .9rem; margin: .6rem 0 .4rem 0; background: var(--surface); }
.jr-source-head { font-size: .88rem; margin-bottom: .35rem; color: var(--text); }
.jr-source-head .ico { color: var(--c, var(--primary)); }
.jr-badge { float: right; font-size: .72rem; background: var(--surface2); color: var(--muted); border-radius: 999px; padding: .1rem .55rem; }
.jr-badge.ok { background: var(--ok-bg); color: var(--ok-fg); }
.jr-bar { height: 4px; background: var(--surface2); border-radius: 99px; overflow: hidden; margin-bottom: .45rem; }
.jr-bar > div { height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent)); }
.jr-extrait { font-size: .8rem; color: var(--muted); line-height: 1.45; }

/* ---------- Tableau de bord ---------- */
.jr-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin-bottom: 1rem; }
@media (max-width: 760px) { .jr-kpis { grid-template-columns: repeat(2, 1fr); } }
.jr-kpi { display: flex; align-items: center; gap: .8rem; background: var(--surface); border: 1px solid var(--border);
    border-top: 4px solid var(--c); border-radius: 12px; padding: .8rem .9rem; box-shadow: var(--shadow); }
.jr-kpi-ico { display: flex; align-items: center; justify-content: center; width: 2.5rem; height: 2.5rem;
    border-radius: 10px; background: var(--b); color: var(--c); flex-shrink: 0; }
.jr-kpi-label { font-size: .75rem; color: var(--muted); line-height: 1.2; }
.jr-kpi-val { font-size: 1.55rem; font-weight: 700; color: var(--text); line-height: 1.2; }
.jr-kpi-sub { font-size: .72rem; color: var(--muted); }
.jr-card-title { display: flex; align-items: center; gap: .5rem; font-weight: 600; color: var(--text); margin-bottom: .7rem; }
.jr-card-title svg { color: var(--primary); }
.jr-hbar { display: grid; grid-template-columns: 11rem 1fr 2.2rem; gap: .6rem; align-items: center; font-size: .85rem;
    margin-bottom: .45rem; color: var(--text); }
.jr-hbar div { background: var(--surface2); height: 10px; border-radius: 99px; overflow: hidden; }
.jr-hbar i { display: block; height: 100%; border-radius: 99px; }
.jr-hbar b { text-align: right; }
.jr-stack { display: flex; height: 14px; border-radius: 99px; overflow: hidden; background: var(--surface2); margin: .2rem 0 .5rem 0; }
.jr-stack i { display: block; height: 100%; }
.jr-legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .8rem; color: var(--muted); margin-bottom: .9rem; }
.jr-legend i { display: inline-block; width: .65rem; height: .65rem; border-radius: 50%; margin-right: .35rem; }
.jr-legend b { color: var(--text); }
.jr-cols { display: flex; align-items: flex-end; gap: .4rem; height: 150px; padding-top: .3rem; }
.jr-col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; min-width: 0; }
.jr-col i { display: block; width: 100%; max-width: 34px; border-radius: 6px 6px 0 0;
    background: linear-gradient(180deg, var(--accent), var(--primary)); }
.jr-col b { font-size: .72rem; color: var(--text); margin-bottom: .15rem; }
.jr-col span { font-size: .68rem; color: var(--muted); margin-top: .25rem; white-space: nowrap; }
.jr-qrow { display: flex; align-items: center; gap: .5rem; padding: .45rem 0; border-bottom: 1px solid var(--border);
    font-size: .85rem; color: var(--text); }
.jr-qrow:last-child { border-bottom: 0; }
.jr-qrow .jr-tag { margin: 0; flex-shrink: 0; }
.jr-qtxt { flex: 1; min-width: 0; line-height: 1.35; }
.jr-qcount { background: var(--surface2); color: var(--text); border-radius: 999px; padding: .05rem .55rem; font-weight: 700;
    font-size: .78rem; flex-shrink: 0; }
.jr-qrow.miss .jr-qtxt { color: var(--low-fg); }
.jr-plan-name { font-weight: 700; color: var(--primary); letter-spacing: .06em; font-size: .8rem; }
.jr-plan-price { font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1.15; }
.jr-plan-price small { font-size: .85rem; font-weight: 500; color: var(--muted); }
</style>
"""

CSS_MINI = """
.stApp [data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] { display: none !important; }
.stApp [data-testid="stSidebar"] .stButton button { justify-content: center !important; padding: .55rem 0 !important; }
.stApp [data-testid="stSidebar"] .stButton button > div { justify-content: center !important; }
[data-testid="stSidebarContent"] { padding: 0 .1rem; }
"""


def construire_css(theme, mini):
    variables = ":root{" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in PALETTES[theme].items())
    variables += "".join(f"--tc-{i}:{c[0]};--tb-{i}:{c[1]};" for i, c in TOPIC_COLORS[theme].items()) + "}"
    largeur = "76px" if mini else "280px"
    return (CSS_TEMPLATE.replace("/*VARS*/", variables).replace("__W__", largeur)
            .replace("/*MINI*/", CSS_MINI if mini else ""))


st.markdown(construire_css(st.session_state["theme"], st.session_state["mini"]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Index, cache et journal d'usage
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement de la base juridique...")
def charger_index(path):
    """Charge l'index FAISS une seule fois et renvoie sa configuration."""
    return rag.load_index(path)


@st.cache_resource
def cache_reponses():
    """Cache partagé des réponses (une même question n'appelle l'IA qu'une fois)."""
    return {}


def interroger(question):
    """Appelle rag.ask() avec cache ; renvoie (résultat, vient_du_cache)."""
    cle = re.sub(r"\s+", " ", question.strip().lower())
    cache = cache_reponses()
    if cle in cache:
        return dict(cache[cle]), True
    debut = time.time()
    res = rag.ask(question)
    res["latency"] = round(time.time() - debut, 1)
    if res.get("answerable") and not res.get("error"):
        cache[cle] = res
    return res, False


def log_event(event):
    """Ajoute un événement anonymisé (aucun identifiant d'utilisateur) au journal d'usage."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        event = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass  # le journal est un bonus : jamais bloquant


def lire_logs():
    if not os.path.exists(LOG_PATH):
        return []
    events = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def detecter_thematique(question):
    """Thématique la plus probable d'une question (mots-clés), sinon « Autre »."""
    q = question.lower()
    scores = {name: sum(1 for kw in t["keywords"] if kw in q) for name, t in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Autre"


def suggestions_liees(question, n=3):
    """Questions de la même thématique (puis des autres) pour poursuivre la conversation."""
    theme = detecter_thematique(question)
    pool = list(TOPICS.get(theme, {}).get("questions", []))
    for name, t in TOPICS.items():
        if name != theme:
            pool += t["questions"]
    seen, out = {question.strip().lower()}, []
    for q in pool:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


def derniere_question(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return None


# ---------------------------------------------------------------------------
# Rappels (callbacks) : exécutés avant le rafraîchissement de la page
# ---------------------------------------------------------------------------
def aller_page(cle):
    st.session_state["page"] = cle


def poser_question(q):
    """Programme une question et bascule sur la page de l'assistant."""
    st.session_state["pending"] = q
    st.session_state["page"] = "assistant"


def nouvelle_conversation():
    st.session_state["messages"] = []
    for k in [k for k in st.session_state if k.startswith("fb_")]:
        del st.session_state[k]


def basculer_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"


def basculer_mini():
    st.session_state["mini"] = not st.session_state["mini"]


def remplir_question_document(texte):
    st.session_state["doc_q"] = texte


def reinitialiser_parcours():
    for i in range(len(CHECKLIST)):
        st.session_state[f"chk_{i}"] = False


def enregistrer_feedback(idx):
    val = st.session_state.get(f"fb_{idx}")
    if val is None:
        return
    msg = st.session_state["messages"][idx]
    log_event({"type": "feedback", "question": msg.get("question", ""),
               "value": "up" if val == 1 else "down"})


# ---------------------------------------------------------------------------
# Affichage : citations, tableaux, figures, sources
# ---------------------------------------------------------------------------
def citations_html(texte):
    """Transforme les citations [1], [2, 3] en pastilles."""
    texte = texte.replace("<", "&lt;")

    def remplacer(m):
        nums = re.split(r"\s*,\s*", m.group(1))
        return "".join(f'<span class="jr-cite">{n}</span>' for n in nums)

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", remplacer, texte)


def nettoyer_extrait(texte):
    """Retire les symboles isolés issus de l'OCR pour un texte plus lisible."""
    return re.sub(r"\s+", " ", re.sub(r"(?:(?<=\s)|^)[|!;'’‘_*~]+(?=\s|$)", " ", texte)).strip()


# Noms de colonnes à utiliser quand l'ingestion n'a pas conservé l'en-tête (typiquement les tableaux
# qui se poursuivent sur plusieurs pages du guide). Clé : (page du PDF, nombre de colonnes affichées).
# Complétez-le au besoin : repérez la page du tableau dans le guide et recopiez son en-tête.
ENTETES = {
    (45, 3): ["Étape", "Description", "Sources d'informations / financement / organes intervenants"],
    (46, 3): ["Étape", "Description", "Délai"],
    (46, 2): ["Description", "Délai"],
    (17, 4): ["N°", "Formalités", "Intervenants", "Dossiers"],
    (18, 4): ["N°", "Formalités", "Intervenants", "Dossiers"],
    (19, 4): ["N°", "Formalités", "Intervenants", "Dossiers"],
    (17, 7): ["N°", "Formalités", "Intervenants", "Dossiers", "SA", "SARL / SUARL", "EI"],
    (18, 7): ["N°", "Formalités", "Intervenants", "Dossiers", "SA", "SARL / SUARL", "EI"],
    (19, 7): ["N°", "Formalités", "Intervenants", "Dossiers", "SA", "SARL / SUARL", "EI"],
    (28, 2): ["Gouvernorat", "Délégations"],
    (58, 2): ["Rubrique", "Détail"],
    (59, 2): ["Rubrique", "Détail"],
    (60, 2): ["Rubrique", "Détail"],
    (66, 2): ["Sigle", "Signification"],
}

PDF_PATH = None   # chemin du guide PDF, renseigné au point d'entrée (affichage de la page d'origine)


class _TableParser(HTMLParser):
    """Lit un tableau HTML : lignes de texte, et indique pour chaque ligne si elle est faite de <th>."""

    def __init__(self):
        super().__init__()
        self.rows, self.th_rows = [], []
        self._row, self._row_th, self._cell = None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row, self._row_th = [], []
        elif tag in ("td", "th"):
            self._cell = []
            if self._row_th is not None:
                self._row_th.append(tag == "th")
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self.th_rows.append(bool(self._row_th) and all(self._row_th))
            self._row, self._row_th = None, None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_JUNK = re.compile(r"^[\W_]+$")  # cellule ne contenant que des symboles
_MOTS_COURTS = {"de", "du", "la", "le", "les", "un", "une", "au", "aux", "en", "et", "ou", "à", "a",
                "d", "l", "par", "sur", "si", "ne", "pas", "se", "sa", "ses", "son", "ce", "n°"}
# Puces que l'OCR lit comme des lettres ou des chiffres : neutres pour la mesure de qualité
_PUCES_OCR = {"i", "ï", "¢", "•", "·", "—", "–", "1", "l", "-"}


def nettoyer_cellule(texte):
    """Nettoie une cellule : symboles OCR isolés, puces lues comme « i », « ï », « ¢ » ou « 1 »."""
    t = nettoyer_extrait(texte)
    t = re.sub(r"^(?:[:•·¢ï—–]|i(?=\s))\s*", "", t)             # puce ou deux-points en tête
    t = re.sub(r"^1\s+(?=[A-ZÉÈÀÂÎ])", "", t)                    # « 1 Les centres » -> puce lue « 1 »
    t = re.sub(r"(?<=[.;:)\w])\s+(?:[ï¢•·]|i)\s+(?=[A-ZÉÈÀÂÎ«(])", " • ", t)   # puces au milieu du texte
    t = re.sub(r"(?<=:)\s+1\s+(?=[A-ZÉÈÀÂÎ])", " • ", t)
    return t.strip()


def lire_tableau(source_html, brut=False):
    """Extrait un tableau propre : (lignes, première ligne faite de <th>).
    Retire cellules parasites OCR, lignes et colonnes vides.
    brut=True conserve le texte tel quel (pour mesurer la qualité de l'OCR)."""
    parser = _TableParser()
    try:
        parser.feed(source_html)
    except Exception:
        return [], False
    if not parser.rows:
        return [], False
    largeur = max(len(r) for r in parser.rows)
    lignes = []
    for r, th in zip(parser.rows, parser.th_rows):
        r = r + [""] * (largeur - len(r))
        r = ["" if _JUNK.match(c) else (c if brut else nettoyer_cellule(c)) for c in r]
        if any(r):
            lignes.append((r, th))
    if not lignes:
        return [], False
    garder = [j for j in range(largeur) if any(r[j] for r, _ in lignes)]
    return [[r[j] for j in garder] for r, _ in lignes], lignes[0][1]


def tableau_vers_lignes(source_html, brut=False):
    return lire_tableau(source_html, brut)[0]


def qualite_tableau(rows):
    """Score entre 0 et 1 : part de mots plausibles (les symboles ou fragments isolés signalent un OCR dégradé)."""
    mots = [m for r in rows for c in r for m in c.split() if m.lower() not in _PUCES_OCR]
    if not mots:
        return 0.0
    mauvais = 0
    for m in mots:
        coeur = m.strip(".,;:()«»\"'’-")
        if not coeur:
            mauvais += 1
        elif re.search(r"[|!_*~^]", coeur):
            mauvais += 1
        elif len(coeur) <= 2 and coeur.lower() not in _MOTS_COURTS and not coeur.isdigit():
            mauvais += 1
    return 1 - mauvais / len(mots)


def tableau_casse(rows):
    """Vrai si la structure du tableau est détruite : texte découpé mot à mot dans des cellules
    en grande partie vides (ex. un encadré ou une liste prise à tort pour un tableau)."""
    cellules = [c for r in rows for c in r]
    if len(cellules) < 6:
        return False
    remplies = [c for c in cellules if c]
    taux = len(remplies) / len(cellules)
    mots = sum(len(c.split()) for c in remplies) / max(len(remplies), 1)
    return (taux < 0.5 and mots < 3) or (len(rows) >= 3 and mots < 1.6)


def ligne_entete(ligne, corps):
    """Vrai si la ligne ressemble à des intitulés de colonnes : cellules courtes, presque toutes remplies,
    et plus brèves que le contenu qui suit."""
    cellules = [c for c in ligne if c]
    if len(cellules) < 2 or len(cellules) < len(ligne) / 2:
        return False
    if any(len(c) > 60 or len(c.split()) > 8 for c in cellules):
        return False
    longueurs = [len(c) for r in corps for c in r if c]
    if longueurs and sum(longueurs) / len(longueurs) < sum(len(c) for c in cellules) / len(cellules):
        return False
    return True


def preparer_entete(rows, entete_balise, page):
    """Retourne (noms de colonnes, lignes de corps, origine) ; origine : detecte, guide ou generique.
    Les noms viennent, dans l'ordre, de l'en-tête lu dans le tableau, de ENTETES (page du guide), puis
    de noms génériques « Colonne n » : un tableau s'affiche toujours avec des intitulés de colonnes."""
    n = len(rows[0])
    if entete_balise or (len(rows) > 1 and ligne_entete(rows[0], rows[1:])):
        noms = [c or f"Colonne {i + 1}" for i, c in enumerate(rows[0])]
        return noms, rows[1:], "detecte"
    noms = ENTETES.get((page, n))
    if noms:
        return list(noms), rows, "guide"
    return [f"Colonne {i + 1}" for i in range(n)], rows, "generique"


def cellule_html(texte):
    """Texte de cellule échappé ; chaque puce « • » repasse à la ligne."""
    return html.escape(texte).replace(" • ", "<br>• ")


def tableau_html(entete, corps):
    th = "".join(f"<th>{html.escape(c)}</th>" for c in entete)
    trs = "".join("<tr>" + "".join(f"<td>{cellule_html(c)}</td>" for c in r) + "</tr>" for r in corps)
    return (f'<div class="jr-table"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{trs}</tbody></table></div>")


# --- Page d'origine du guide (repli quand le tableau extrait est illisible) ---------------------
def trouver_pdf(nom_config=""):
    """Cherche le PDF du guide : JURIA_PDF_PATH, nom indiqué par l'index, ou juria-source.pdf."""
    noms = [n for n in (os.environ.get("JURIA_PDF_PATH"), nom_config, "juria-source.pdf") if n]
    dossiers = [BASE_DIR, os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "assets"),
                os.getcwd(), INDEX_DIR, os.path.dirname(INDEX_DIR.rstrip("/\\")) or "."]
    for n in noms:
        if os.path.isabs(n) and os.path.isfile(n):
            return n
        for d in dossiers:
            chemin = os.path.join(d, os.path.basename(n))
            if os.path.isfile(chemin):
                return chemin
    return None


def _moteur_pdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            return None


@st.cache_data(show_spinner=False, max_entries=48)
def image_page_pdf(chemin, page, dpi=110):
    """Page du PDF rendue en PNG (octets) ; vide si impossible."""
    moteur = _moteur_pdf()
    if moteur is None:
        return b""
    try:
        with moteur.open(chemin) as doc:
            if not 1 <= page <= len(doc):
                return b""
            return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
    except Exception:
        return b""


def page_guide_disponible():
    return bool(PDF_PATH) and _moteur_pdf() is not None


def afficher_page_guide(page):
    octets = image_page_pdf(PDF_PATH, int(page))
    if not octets:
        st.caption("Page du guide indisponible.")
        return
    afficher_image(base64.b64encode(octets).decode(), f"Guide, page {page}")


def _corps_tableau(rows, entete_balise, page, cle, degrade):
    """Tableau avec noms de colonnes, résumé et export CSV."""
    entete, corps, origine = preparer_entete(rows, entete_balise, page)
    st.markdown(tableau_html(entete, corps), unsafe_allow_html=True)
    tampon = io.StringIO()
    csv.writer(tampon).writerows([entete] + corps)
    notes = ""
    if origine == "generique":
        notes += " · en-tête d'origine non conservé"
    if degrade:
        notes += " · qualité de lecture réduite"
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.caption(f"{len(corps)} ligne(s) · {len(entete)} colonne(s){notes}")
    c2.download_button("CSV", data=tampon.getvalue().encode("utf-8-sig"),
                       file_name="tableau_juria.csv", mime="text/csv",
                       key=f"csv_{cle}", **ic("download"))


def _alerte(titre, texte):
    st.markdown(f'<div class="jr-callout">{svg("alert", 18)}<div><b>{titre}</b> {texte}</div></div>',
                unsafe_allow_html=True)


def afficher_tableau(source_html, cle, pages=None):
    """Affiche un tableau lisible, avec noms de colonnes. Si la structure est perdue ou l'OCR dégradé,
    prévient et propose la page d'origine du guide. N'utilise aucun expander (appelable dans un expander)."""
    rows, entete_balise = lire_tableau(source_html)
    if len(rows) < 1 or sum(len(r) for r in rows) < 2:
        st.caption(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", source_html)).strip()[:1500])
        return

    page = int(pages[0]) if pages else None
    guide = bool(page) and page_guide_disponible()
    casse = tableau_casse(rows)
    degrade = qualite_tableau(lire_tableau(source_html, brut=True)[0]) < SEUIL_QUALITE_TABLEAU

    if casse:
        _alerte("Tableau mal reconnu.",
                "Le texte a été découpé en cellules dans le document d'origine : la structure est "
                "perdue. " + (f"Voici la page {page} du guide." if guide else
                              "Consultez le guide à la page indiquée pour lire le tableau."))
        if guide:
            afficher_page_guide(page)
        elif page:
            st.caption("Pour afficher la page ici, placez le PDF du guide à côté de app.py "
                       "et installez pymupdf.")
        if not guide or st.toggle("Voir le texte extrait (structure perdue)", key=f"txt_{cle}"):
            _corps_tableau(rows, entete_balise, page, cle, degrade)
        return

    if degrade:
        _alerte("Lecture partielle du tableau.",
                "Le document d'origine est un scan et la reconnaissance de texte (OCR) a produit des "
                "erreurs" + (f" : comparez avec la page {page} du guide." if page else "."))
    _corps_tableau(rows, entete_balise, page, cle, degrade)
    if guide and st.toggle(f"Voir la page {page} du guide", key=f"pg_{cle}"):
        afficher_page_guide(page)


def afficher_image(octets_b64, legende):
    donnees = base64.b64decode(octets_b64)
    for options in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(donnees, caption=legende, **options)
            return
        except Exception:  # selon la version de Streamlit, le paramètre diffère
            continue


def associer_medias(res, rattacher=True):
    """Rattache chaque tableau / figure de la réponse à la source dont il provient.

    rag.ask() renvoie les médias dans res["tables"] et res["images"], dans l'ordre des sources de
    type « table » / « image » (celles qui sont citées, ou toutes si rien n'est cité). Quand les
    effectifs ne concordent pas, le média est affiché à part plutôt que rattaché au mauvais passage.
    Retourne (médias par index de source, tableaux libres, figures libres)."""
    sources = res.get("sources", [])
    a_cite = any(s["cited"] for s in sources)
    rattaches, libres = {}, {}
    for genre, champ in (("table", "tables"), ("image", "images")):
        medias = list(res.get(champ) or [])
        if rattacher:
            candidats = [i for i, s in enumerate(sources) if s["type"] == genre]
            cites = [i for i in candidats if sources[i]["cited"] or not a_cite]
            for groupe in (cites, candidats):
                if medias and groupe and len(groupe) == len(medias):
                    for i, m in zip(groupe, medias):
                        rattaches.setdefault(i, {})[genre] = m
                    medias = []
                    break
        libres[genre] = medias
    return rattaches, libres["table"], libres["image"]


def carte_source(n, s, media, cle):
    """Une source : en-tête (numéro, type, page, statut), proximité, extrait et média associé."""
    icone, nom, couleur = TYPES_FR.get(s["type"], TYPES_FR["text"])
    pages = ", ".join(map(str, s["pages"])) or "?"
    badge = ('<span class="jr-badge ok">Citée</span>' if s["cited"]
             else '<span class="jr-badge">Consultée</span>')
    largeur = max(0, min(100, int(s["score"] * 100)))
    montre = media is not None and (
        (s["type"] == "table" and st.session_state.get("show_tables", True))
        or (s["type"] == "image" and st.session_state.get("show_images", True)))
    extrait = ("" if montre else
               f'<div class="jr-extrait">{html.escape(nettoyer_extrait(s["extrait"]))}</div>')
    st.markdown(
        f'<div class="jr-source" style="--c:{couleur}"><div class="jr-source-head">'
        f'<span class="jr-cite">{n}</span> <span class="ico">{svg(icone)}</span> <b>{nom}</b> · page {pages}{badge}</div>'
        f'<div class="jr-bar"><div style="width:{largeur}%"></div></div>{extrait}</div>',
        unsafe_allow_html=True)
    if montre and s["type"] == "table":
        afficher_tableau(media, f"{cle}_s{n}", s["pages"])
    elif montre and s["type"] == "image":
        afficher_image(media, f"Figure · source {n}, page {pages}")


def afficher_sources(res, cle, medias):
    """Sources en deux onglets : citées (utilisées dans la réponse) et consultées (simple contexte)."""
    sources = res.get("sources", [])
    if not sources:
        return
    numerotees = list(enumerate(sources, start=1))
    citees = sorted((t for t in numerotees if t[1]["cited"]), key=lambda t: -t[1]["score"])
    consultees = sorted((t for t in numerotees if not t[1]["cited"]), key=lambda t: -t[1]["score"])
    a_media = any(i - 1 in medias for i, _ in citees)
    with st.expander(f"Sources ({len(sources)})", expanded=a_media):
        onglet_c, onglet_k = st.tabs([f"Citées ({len(citees)})", f"Consultées ({len(consultees)})"])
        for onglet, groupe, aide in (
                (onglet_c, citees, "Passages sur lesquels la réponse s'appuie : elle y renvoie par les "
                                   "pastilles numérotées du texte."),
                (onglet_k, consultees, "Passages retrouvés par la recherche et transmis à l'IA comme "
                                       "contexte, mais que la réponse ne cite pas : moins directement "
                                       "utiles, ils servent à contrôler.")):
            with onglet:
                st.caption(aide)
                if not groupe:
                    st.caption("Aucun passage dans cette catégorie.")
                for n, s in groupe:
                    media = medias.get(n - 1, {}).get(s["type"])
                    carte_source(n, s, media, cle)
        st.caption("La barre indique la proximité entre la question et le passage retrouvé.")


def niveau_pertinence(res):
    """Indicateur indicatif de pertinence, d'après la meilleure source citée."""
    sources = res.get("sources", [])
    scores = [s["score"] for s in sources if s["cited"]] or [s["score"] for s in sources]
    if not scores:
        return None
    best = max(scores)
    if best >= 0.6:
        return "Pertinence élevée", "high"
    if best >= 0.45:
        return "Pertinence moyenne", "mid"
    return "Pertinence faible", "low"


def animer_texte(zone, texte):
    """Effet de frappe (la réponse est déjà complète, c'est purement visuel)."""
    morceaux = re.findall(r"\S+\s*", texte)
    delai = min(0.03, 1.8 / max(len(morceaux), 1))
    acc = ""
    for m in morceaux:
        acc += m
        zone.markdown(citations_html(acc) + " ▌", unsafe_allow_html=True)
        time.sleep(delai)
    zone.markdown(citations_html(texte), unsafe_allow_html=True)


def afficher_sans_reponse(res, cle, question):
    raison = res.get("reason")
    if raison == "no_results":
        detail = "Aucun passage de la base n'est assez proche de votre question."
    elif raison == "no_text":
        detail = res.get("answer", "")
    else:
        detail = "La base documentaire ne contient pas de réponse fiable à cette question."
    st.markdown(
        f'<div class="jr-empty"><b>{svg("search", 18)} Information introuvable</b>{html.escape(detail)}<br>'
        "Je préfère ne pas répondre au hasard. Essayez de reformuler avec des termes plus précis, "
        "ou consultez un avocat pour ce point.</div>", unsafe_allow_html=True)
    if question:
        st.caption("Pour continuer, vous pouvez essayer :")
        for i, q in enumerate(suggestions_liees(question, 2)):
            st.button(q, key=f"alt_{cle}_{i}", on_click=poser_question, args=(q,), **STRETCH)


def afficher_reponse(res, cle, idx=None, derniere=False, question="", animer=False, theme=""):
    """Affiche un résultat de rag.ask() : réponse, alerte, sources (avec tableaux et figures), retour."""
    if res.get("error"):
        st.error(res["answer"])
        st.caption("Vérifiez la clé API et la connexion, puis réessayez.")
        if question:
            st.button("Réessayer", key=f"retry_{cle}", on_click=poser_question, args=(question,),
                      **ic("refresh"))
        return
    if not res.get("answerable", True):
        afficher_sans_reponse(res, cle, question)
        return

    texte = res["answer"]
    zone = st.empty()
    if animer and st.session_state.get("animation", True):
        animer_texte(zone, texte)
    else:
        zone.markdown(citations_html(texte), unsafe_allow_html=True)

    chips = []
    if theme in TOPICS:
        chips.append(f'<span class="jr-tag" style="{topic_style(theme)}">{svg(TOPICS[theme]["icon"], 13)}{theme}</span>')
    pert = niveau_pertinence(res)
    if pert:
        chips.append(f'<span class="jr-chip {pert[1]}">{svg("gauge", 13)}{pert[0]}</span>')
    if res.get("sources"):
        chips.append(f'<span class="jr-chip">{svg("library", 13)}{len(res["sources"])} sources</span>')
    if res.get("tables"):
        chips.append(f'<span class="jr-chip">{svg("table", 13)}{len(res["tables"])} tableau(x)</span>')
    if res.get("images"):
        chips.append(f'<span class="jr-chip">{svg("image", 13)}{len(res["images"])} figure(s)</span>')
    if res.get("latency") is not None:
        chips.append(f'<span class="jr-chip">{svg("clock", 13)}{str(res["latency"]).replace(".", ",")} s</span>')
    if chips:
        st.markdown(f'<div class="jr-meta">{"".join(chips)}</div>', unsafe_allow_html=True)

    if re.search(r"avocat", texte, re.IGNORECASE):
        st.markdown(f'<div class="jr-callout">{svg("scale", 18)}<div><b>Enjeu important.</b> JurIA donne de '
                    "l'information juridique, pas un conseil : pour sécuriser votre décision, faites-la "
                    "valider par un avocat.</div></div>", unsafe_allow_html=True)

    # Tableaux et figures : rattachés à leur source ; ceux qu'on ne peut pas rattacher sont montrés ici.
    montrer_sources = st.session_state.get("show_sources", True)
    medias, tables_libres, images_libres = associer_medias(res, rattacher=montrer_sources)
    if tables_libres and st.session_state.get("show_tables", True):
        for i, t in enumerate(tables_libres):
            with st.expander(f"Tableau {i + 1}", expanded=(i == 0)):
                afficher_tableau(t, f"{cle}_t{i}")
    if images_libres and st.session_state.get("show_images", True):
        with st.expander(f"Figures ({len(images_libres)})", expanded=True):
            for i, img in enumerate(images_libres):
                afficher_image(img, f"Figure {i + 1}")
    if montrer_sources:
        afficher_sources(res, cle, medias)

    if idx is not None and hasattr(st, "feedback"):
        st.caption("Cette réponse vous a-t-elle été utile ?")
        st.feedback("thumbs", key=f"fb_{idx}", on_change=enregistrer_feedback, args=(idx,))
    if derniere and question:
        st.caption("Pour aller plus loin")
        for i, q in enumerate(suggestions_liees(question)):
            st.button(q, key=f"sug_{cle}_{i}", on_click=poser_question, args=(q,), **STRETCH)


# ---------------------------------------------------------------------------
# Page : assistant
# ---------------------------------------------------------------------------
AVATAR_IA = ICON_PATH


def afficher_accueil(config):
    counts = config.get("counts", {})
    st.markdown(
        '<div class="jr-hero"><h1>Assistant juridique</h1>'
        "<p>Posez vos questions en langage courant. Les réponses sont claires, sourcées et adaptées "
        "au contexte tunisien.</p>"
        f'<span class="jr-feature">{svg("library")}Réponses citant leurs sources</span>'
        f'<span class="jr-feature">{svg("table")}Tableaux et figures</span>'
        f'<span class="jr-feature">{svg("file-text")}Analyse de vos documents</span>'
        f'<span class="jr-feature">{svg("shield")}Startups et PME tunisiennes</span></div>',
        unsafe_allow_html=True)
    if counts:
        c1, c2, c3 = st.columns(3)
        c1.metric("Passages de texte", counts.get("text", 0))
        c2.metric("Tableaux", counts.get("table", 0))
        c3.metric("Figures", counts.get("image", 0))
        st.caption("Base documentaire : « Guide du jeune entrepreneur » (ONUDI, projet Mashrou3i, 2014). "
                   "Certains montants et taux ont pu évoluer depuis : vérifiez-les auprès des organismes.")
    st.markdown('<div class="jr-title" style="margin-top:1.2rem">Par où commencer ?</div>'
                '<div class="jr-subtitle">Choisissez une thématique ou saisissez votre question ci-dessous.</div>',
                unsafe_allow_html=True)
    noms = list(TOPICS)
    for ligne in range(0, len(noms), 3):
        cols = st.columns(3)
        for col, nom in zip(cols, noms[ligne:ligne + 3]):
            t = TOPICS[nom]
            with col.container(border=True):
                st.markdown(f'<div class="jr-thead" style="{topic_style(nom)}"><span class="ico">'
                            f'{svg(t["icon"], 22)}</span><div><b>{nom}</b><small>{t["desc"]}</small></div></div>',
                            unsafe_allow_html=True)
                for i, q in enumerate(t["questions"]):
                    st.button(q, key=f"home_{nom}_{i}", on_click=poser_question, args=(q,), **STRETCH)


def traiter_question(question):
    """Pose la question à JurIA et affiche la réponse dans la conversation."""
    messages = st.session_state["messages"]
    question = question.strip()[:MAX_QUESTION_CHARS]

    # Relance courte (« et pour une SA ? ») : on la rattache à la question précédente
    requete = question
    precedente = derniere_question(messages)
    if (st.session_state.get("contexte", True) and precedente
            and len(question.split()) <= MAX_MOTS_SUIVI):
        requete = f"{precedente} {question}"

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATAR_IA):
        with st.status("JurIA analyse votre question…", expanded=False) as status:
            st.write("Recherche des passages pertinents dans la base juridique")
            res, depuis_cache = interroger(requete)
            st.write("Rédaction de la réponse")
            status.update(label="Réponse prête", state="complete")
        if depuis_cache:
            res["latency"] = 0.0
        idx = len(messages)
        theme = detecter_thematique(requete)
        messages.append({"role": "assistant", "result": res, "question": question, "topic": theme})
        if not res.get("error"):
            log_event({"type": "question", "question": question, "topic": theme,
                       "answerable": bool(res.get("answerable")),
                       "latency": res.get("latency") if not depuis_cache else None})
        afficher_reponse(res, f"m{idx}", idx=idx, derniere=True, question=question, animer=True,
                         theme=theme)


def page_assistant(config):
    messages = st.session_state["messages"]
    if not messages:
        afficher_accueil(config)
    for i, m in enumerate(messages):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=AVATAR_IA):
                afficher_reponse(m["result"], f"m{i}", idx=i, derniere=(i == len(messages) - 1),
                                 question=m["question"], theme=m.get("topic", ""))
    question = st.chat_input("Posez votre question juridique…", max_chars=MAX_QUESTION_CHARS)
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        traiter_question(question)


# ---------------------------------------------------------------------------
# Page : analyse de document
# ---------------------------------------------------------------------------
def page_document():
    st.markdown('<div class="jr-title">Analyser un document</div>'
                '<div class="jr-subtitle">Déposez un contrat, un formulaire ou un courrier : JurIA le lit '
                "et le confronte à sa base juridique.</div>", unsafe_allow_html=True)
    fichier = st.file_uploader("Document (PDF avec texte, ou photo PNG/JPG)",
                               type=["pdf", "png", "jpg", "jpeg"])
    st.caption("Exemples de demandes")
    cols = st.columns(len(DOC_PROMPTS))
    for col, p in zip(cols, DOC_PROMPTS):
        col.button(p, key=f"dp_{p}", on_click=remplir_question_document, args=(p,), **STRETCH)
    question = st.text_area("Que voulez-vous savoir ? (facultatif)", key="doc_q", height=90,
                            max_chars=MAX_QUESTION_CHARS)
    trop_gros = bool(fichier) and fichier.size > MAX_UPLOAD_MB * 1024 * 1024
    if trop_gros:
        st.error(f"Fichier trop volumineux (maximum {MAX_UPLOAD_MB} Mo).")
    if fichier and fichier.type.startswith("image/"):
        st.image(fichier.getvalue(), caption=fichier.name, width=320)

    if st.button("Analyser le document", type="primary", disabled=(fichier is None or trop_gros),
                 **ic("search")):
        if not hasattr(rag, "ask_document"):
            st.error("La fonction d'analyse de documents n'est pas disponible : mettez rag.py à jour.")
        else:
            with st.status("Analyse en cours…", expanded=False) as status:
                debut = time.time()
                res = rag.ask_document(question, fichier.getvalue(), fichier.type)
                res["latency"] = round(time.time() - debut, 1)
                status.update(label="Analyse terminée", state="complete")
            st.session_state["doc_result"] = (fichier.name, res)
            if not res.get("error"):
                log_event({"type": "document", "topic": "Analyse de document",
                           "answerable": bool(res.get("answerable"))})

    if "doc_result" in st.session_state:
        nom, res = st.session_state["doc_result"]
        st.divider()
        st.markdown(f"**Résultat pour :** `{nom}`")
        with st.container(border=True):
            afficher_reponse(res, "doc")
        st.caption("Seuls les PDF contenant du texte sont lus ; pour un scan, envoyez une photo. "
                   "Les fichiers ne sont pas conservés par l'application.")


# ---------------------------------------------------------------------------
# Page : parcours de création
# ---------------------------------------------------------------------------
def page_checklist():
    st.markdown('<div class="jr-title">Parcours de création de société</div>'
                '<div class="jr-subtitle">Suivez les étapes une par une et demandez à JurIA le détail '
                "des documents de chacune.</div>", unsafe_allow_html=True)
    faites = sum(1 for i in range(len(CHECKLIST)) if st.session_state.get(f"chk_{i}"))
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    c1.progress(faites / len(CHECKLIST), text=f"{faites} étape(s) sur {len(CHECKLIST)} terminée(s)")
    c2.button("Réinitialiser", on_click=reinitialiser_parcours, key="chk_reset", **ic("restart_alt"),
              **STRETCH)
    for i, (etape, organisme, question) in enumerate(CHECKLIST):
        with st.container(border=True):
            a, b, c = st.columns([0.7, 5, 2.6], vertical_alignment="center")
            a.checkbox(f"Étape {i + 1}", key=f"chk_{i}", label_visibility="collapsed")
            b.markdown(f"**{i + 1}. {etape}**  \n:gray[Auprès de : {organisme}]")
            c.button("Détails avec JurIA", key=f"chk_btn_{i}", on_click=poser_question,
                     args=(question,), **ic("chat"), **STRETCH)
    if faites == len(CHECKLIST):
        st.success("Toutes les étapes sont cochées. Vérifiez auprès des administrations que votre "
                   "dossier est complet.")
    st.caption("Étapes issues du guide indexé (formalités de création des sociétés). La coche est un "
               "aide-mémoire local : elle n'est pas enregistrée.")


# ---------------------------------------------------------------------------
# Page : tableau de bord
# ---------------------------------------------------------------------------
def kpi_html(libelle, valeur, icone, couleur, fond, detail=""):
    """Carte d'indicateur : couleur d'accent (variable CSS), icône, valeur et précision."""
    return (f'<div class="jr-kpi" style="--c:var({couleur});--b:var({fond})">'
            f'<div class="jr-kpi-ico">{svg(icone, 20)}</div><div>'
            f'<div class="jr-kpi-label">{html.escape(libelle)}</div>'
            f'<div class="jr-kpi-val">{html.escape(str(valeur))}</div>'
            f'<div class="jr-kpi-sub">{html.escape(detail)}</div></div></div>')


def titre_carte(icone, texte):
    return f'<div class="jr-card-title">{svg(icone, 16)}{html.escape(texte)}</div>'


def barres_html(lignes):
    """Histogramme horizontal ; lignes = [(libellé, valeur, couleur CSS)]. Compatible clair/sombre."""
    maxi = max((v for _, v, _ in lignes), default=1) or 1
    return "".join(
        f'<div class="jr-hbar"><span>{html.escape(str(k))}</span>'
        f'<div><i style="width:{100 * v / maxi:.0f}%;background:{c}"></i></div><b>{v}</b></div>'
        for k, v, c in lignes)


def barre_empilee(parts):
    """Barre segmentée + légende ; parts = [(libellé, valeur, couleur CSS)]."""
    total = sum(v for _, v, _ in parts) or 1
    barre = "".join(f'<i style="width:{100 * v / total:.1f}%;background:{c}"></i>' for _, v, c in parts if v)
    legende = "".join(f'<span><i style="background:{c}"></i>{html.escape(k)} <b>{v}</b></span>'
                      for k, v, c in parts)
    return f'<div class="jr-stack">{barre}</div><div class="jr-legend">{legende}</div>'


def colonnes_html(serie):
    """Histogramme vertical par jour ; serie = [(étiquette, valeur)]."""
    maxi = max((v for _, v in serie), default=1) or 1
    cols = "".join(f'<div class="jr-col"><b>{v}</b><i style="height:{max(4, round(100 * v / maxi))}px"></i>'
                   f"<span>{html.escape(e)}</span></div>" for e, v in serie)
    return f'<div class="jr-cols">{cols}</div>'


def ligne_question(texte, theme, compte=None, manquante=False):
    tag = (f'<span class="jr-tag" style="{topic_style(theme)}">{html.escape(str(theme))}</span>')
    nb = f'<span class="jr-qcount">{compte}</span>' if compte is not None else ""
    return (f'<div class="jr-qrow{" miss" if manquante else ""}">{tag}'
            f'<span class="jr-qtxt">{html.escape(texte)}</span>{nb}</div>')


def page_dashboard():
    st.markdown('<div class="jr-title">Tableau de bord</div>'
                '<div class="jr-subtitle">Usage anonymisé de JurIA : quelles questions reviennent, '
                "lesquelles restent sans réponse.</div>", unsafe_allow_html=True)
    events = lire_logs()
    questions = [e for e in events if e.get("type") == "question"]
    feedbacks = [e for e in events if e.get("type") == "feedback"]
    if not questions:
        st.info("Aucune donnée pour l'instant : posez quelques questions dans l'assistant, "
                "elles apparaîtront ici.")
        return
    df = pd.DataFrame(questions)
    df["answerable"] = df["answerable"].fillna(False).astype(bool)
    if "topic" not in df:
        df["topic"] = "Autre"
    df["topic"] = df["topic"].fillna("Autre")
    n = len(df)
    n_ok = int(df["answerable"].sum())
    n_ko = n - n_ok
    ups = sum(1 for f in feedbacks if f.get("value") == "up")
    downs = sum(1 for f in feedbacks if f.get("value") == "down")
    latences = (pd.to_numeric(df["latency"], errors="coerce").dropna()
                if "latency" in df else pd.Series(dtype=float))

    # --- Indicateurs clés ---
    kpis = [
        kpi_html("Questions posées", n, "message", "--primary", "--soft",
                 f"{df['question'].nunique()} questions distinctes"),
        kpi_html("Taux de réponse", f"{round(100 * n_ok / n)} %", "gauge", "--ok-fg", "--ok-bg",
                 f"{n_ko} sans réponse"),
        kpi_html("Temps moyen", f"{latences.mean():.1f} s" if len(latences) else "n/a", "clock",
                 "--mid-fg", "--mid-bg", f"max {latences.max():.1f} s" if len(latences) else "hors cache"),
        kpi_html("Satisfaction", f"{round(100 * ups / (ups + downs))} %" if ups + downs else "n/a",
                 "thumb", "--tc-accomp", "--tb-accomp", f"{ups + downs} avis"),
    ]
    st.markdown(f'<div class="jr-kpis">{"".join(kpis)}</div>', unsafe_allow_html=True)

    # --- Thématiques et qualité des réponses ---
    g1, g2 = st.columns([3, 2])
    with g1, st.container(border=True):
        st.markdown(titre_carte("library", "Thématiques les plus demandées"), unsafe_allow_html=True)
        compte = df["topic"].value_counts()
        st.markdown(barres_html([(k, int(v), f"var(--tc-{topic_id(k)})") for k, v in compte.items()]),
                    unsafe_allow_html=True)
    with g2, st.container(border=True):
        st.markdown(titre_carte("gauge", "Qualité des réponses"), unsafe_allow_html=True)
        st.markdown(barre_empilee([("Répondues", n_ok, "var(--ok-fg)"),
                                   ("Sans réponse", n_ko, "var(--low-fg)")]), unsafe_allow_html=True)
        if ups + downs:
            st.markdown(barre_empilee([("Utiles", ups, "var(--ok-fg)"),
                                       ("Pas utiles", downs, "var(--mid-fg)")]), unsafe_allow_html=True)
        else:
            st.caption("Aucun avis pour l'instant (pouces sous les réponses).")

    # --- Activité par jour ---
    if "ts" in df:
        par_jour = df["ts"].astype(str).str[:10].value_counts().sort_index().tail(JOURS_ACTIVITE)
        serie = [(f"{d[8:10]}/{d[5:7]}", int(v)) for d, v in par_jour.items()]
        with st.container(border=True):
            st.markdown(titre_carte("clock", f"Activité des {JOURS_ACTIVITE} derniers jours actifs"),
                        unsafe_allow_html=True)
            st.markdown(colonnes_html(serie), unsafe_allow_html=True)

    # --- Questions fréquentes et questions sans réponse ---
    q1, q2 = st.columns(2)
    with q1, st.container(border=True):
        st.markdown(titre_carte("message", "Questions les plus fréquentes"), unsafe_allow_html=True)
        top = (df.groupby("question").agg(nb=("question", "size"), theme=("topic", "first"))
               .sort_values("nb", ascending=False).head(8))
        st.markdown("".join(ligne_question(q, r["theme"], int(r["nb"])) for q, r in top.iterrows()),
                    unsafe_allow_html=True)
    with q2, st.container(border=True):
        st.markdown(titre_carte("alert", "Questions sans réponse"), unsafe_allow_html=True)
        sans = df[~df["answerable"]].drop_duplicates("question").head(10)
        if len(sans):
            st.markdown("".join(ligne_question(r["question"], r["topic"], manquante=True)
                                for _, r in sans.iterrows()), unsafe_allow_html=True)
            st.caption("Sujets à ajouter à la base documentaire.")
        else:
            st.success("Toutes les questions ont reçu une réponse.")

    st.download_button("Exporter le journal (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="usage_juria.csv", mime="text/csv", **ic("download"))
    st.caption("Seules la question, la thématique et le résultat sont enregistrés, sans identifiant "
               "d'utilisateur.")


# ---------------------------------------------------------------------------
# Page : offres, feuille de route, responsabilité
# ---------------------------------------------------------------------------
def page_about():
    st.markdown('<div class="jr-title">À propos de JurIA</div>'
                '<div class="jr-subtitle">Modèle économique, évolutions prévues et engagements.</div>',
                unsafe_allow_html=True)
    t_offres, t_route, t_resp = st.tabs(["Offres", "Feuille de route", "Responsabilité"])

    with t_offres:
        st.caption("Modèle SaaS par abonnement mensuel. Contenu des offres indicatif, à finaliser.")
        plans = [
            ("FREE", "0", ["Quelques questions par mois", "Réponses sourcées"]),
            ("STARTER", "49", ["Plus de questions", "Analyse de documents", "Historique"]),
            ("BUSINESS", "99", ["Suivi de conformité", "Rappels d'échéances", "Support"]),
            ("PREMIUM", "199", ["Tout Business", "Avis d'avocat (option)", "Support prioritaire"]),
        ]
        for col, (nom, prix, avantages) in zip(st.columns(4), plans):
            with col.container(border=True):
                st.markdown(f'<div class="jr-plan-name">{nom}</div>'
                            f'<div class="jr-plan-price">{prix} DT <small>/ mois</small></div>',
                            unsafe_allow_html=True)
                for a in avantages:
                    st.markdown(f":material/check: {a}")
        st.caption("Organisations d'appui (incubateurs, programmes) : licences annuelles et tableau de bord "
                   "des questions les plus posées par cohorte.")

    with t_route:
        etapes = [
            ("Enrichissement juridique", "Prototype", "Base actuelle : le Guide du jeune entrepreneur "
             "(ONUDI, 2014). Extension prévue : fiscal, social, commercial, propriété intellectuelle."),
            ("Intelligence documentaire", "Prototype", "Analyse de documents disponible ; détection "
             "avancée des clauses à risque à venir."),
            ("Accessibilité multilingue", "À venir", "Prise en charge de l'arabe et du français."),
            ("Veille juridique intelligente", "À venir", "Alertes automatiques sur les changements "
             "législatifs."),
            ("Expansion", "À venir", "Partenariats avec des professionnels du droit, autres marchés "
             "francophones et arabophones."),
        ]
        for i, (titre, statut, desc) in enumerate(etapes, start=1):
            with st.container(border=True):
                a, b = st.columns([4, 1], vertical_alignment="center")
                a.markdown(f"**{i}. {titre}**  \n:gray[{desc}]")
                b.markdown(":green-badge[Prototype]" if statut == "Prototype" else ":gray-badge[À venir]")

    with t_resp:
        st.markdown("**JurIA informe et accompagne, mais ne remplace jamais un avocat.**")
        rows = [["Risque", "Mesure de prévention"],
                ["Hallucinations de l'IA", "RAG + sources vérifiables"],
                ["Textes obsolètes", "Mise à jour du corpus"],
                ["Fuite de données", "Chiffrement et contrôle d'accès"],
                ["Mauvaise interprétation juridique", "Orientation vers un avocat"],
                ["Biais de l'IA", "Tests et contrôle des réponses"]]
        st.markdown(tableau_html(rows[0], rows[1:]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barre latérale (réductible en icônes) et exports
# ---------------------------------------------------------------------------
def exporter_conversation():
    """Conversation au format Markdown (pour archivage ou partage)."""
    lignes = ["# Conversation JurIA", f"_Exportée le {datetime.now():%d/%m/%Y à %H:%M}_", ""]
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            lignes += [f"## {m['content']}", ""]
        else:
            res = m["result"]
            lignes += [res.get("answer", ""), ""]
            for n, s in enumerate(res.get("sources", []), start=1):
                if s["cited"]:
                    pages = ", ".join(map(str, s["pages"])) or "?"
                    lignes.append(f"- [{n}] {s['type']}, page(s) {pages}")
            lignes.append("")
    lignes += ["---", "_JurIA fournit de l'information juridique, pas un conseil. "
               "Consultez un avocat pour les enjeux importants._"]
    return "\n".join(lignes)


def barre_laterale(config):
    mini = st.session_state["mini"]
    sombre = st.session_state["theme"] == "dark"
    with st.sidebar:
        if not mini:
            _, c_reduire = st.columns([5, 1], gap="small")
            c_reduire.button("Réduire le menu", key="btn_mini_top", help="Réduire le menu",
                             on_click=basculer_mini, icon=":material/left_panel_close:", **STRETCH)

        # Logo : emblème seul en mode réduit, logo complet sinon (repli : nom en texte)
        uri = image_uri(ICON_PATH if mini else LOGO_PATH)
        if uri:
            st.markdown(f'<div class="jr-logo{" mini" if mini else ""}"><img src="{uri}" alt="JurIA"></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="jr-wordmark">' + ("J" if mini else "Jur<span>IA</span>") + "</div>",
                        unsafe_allow_html=True)

        for cle, libelle, icone in NAV:
            st.button(libelle, key=f"nav_{cle}", help=libelle,
                      type="primary" if st.session_state["page"] == cle else "secondary",
                      on_click=aller_page, args=(cle,), icon=f":material/{icone}:", **STRETCH)

        if not mini:
            st.markdown('<div class="jr-side-title">Affichage</div>', unsafe_allow_html=True)
        icone_theme = ":material/light_mode:" if sombre else ":material/dark_mode:"
        icone_mini = ":material/left_panel_open:" if mini else ":material/left_panel_close:"
        if mini:
            st.button("Thème", key="btn_theme", help="Basculer clair / sombre", on_click=basculer_theme,
                      icon=icone_theme, **STRETCH)
            st.button("Agrandir le menu", key="btn_mini", help="Agrandir le menu", on_click=basculer_mini,
                      icon=icone_mini, **STRETCH)
        else:
            st.button("Clair" if sombre else "Sombre", key="btn_theme", help="Basculer clair / sombre",
                      on_click=basculer_theme, icon=icone_theme, **STRETCH)

        a_des_messages = bool(st.session_state["messages"])
        if not mini and a_des_messages:
            st.markdown('<div class="jr-side-title">Conversation</div>', unsafe_allow_html=True)
        if a_des_messages:
            st.button("Nouvelle conversation", key="new_conv", help="Nouvelle conversation",
                      on_click=nouvelle_conversation, **ic("add_comment"), **STRETCH)
            if not mini:
                st.download_button("Exporter en Markdown", data=exporter_conversation(),
                                   file_name="conversation_juria.md", mime="text/markdown",
                                   **ic("download"), **STRETCH)

        if not mini:
            recentes = [m["content"] for m in st.session_state["messages"] if m["role"] == "user"][-4:]
            if recentes:
                st.markdown('<div class="jr-side-title">Récentes</div>', unsafe_allow_html=True)
                for i, q in enumerate(reversed(recentes)):
                    st.button(q if len(q) < 30 else q[:28] + "…", key=f"hist_{i}", help=q,
                              on_click=poser_question, args=(q,), **ic("history"), **STRETCH)
            with st.expander("Préférences"):
                st.toggle("Tableaux", value=True, key="show_tables")
                st.toggle("Figures", value=True, key="show_images")
                st.toggle("Sources", value=True, key="show_sources")
                st.toggle("Effet de frappe", value=True, key="animation")
                st.toggle("Relances contextuelles", value=True, key="contexte",
                          help="Une question courte est rattachée à la précédente.")
            st.markdown('<div class="jr-side-note">JurIA fournit de l\'information juridique, pas un '
                        "conseil. Pour un enjeu important, consultez un avocat.<br>"
                        f"<small>Base : {html.escape(str(config.get('source', '?')))} · "
                        f"{config.get('indexed', '?')} éléments</small></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
try:
    CONFIG = charger_index(INDEX_DIR)
except FileNotFoundError:
    st.error(f"Index introuvable dans « {INDEX_DIR} ». Copiez-y les fichiers produits par ingest.py "
             "(index.faiss, index.pkl, originals.json, config.json).")
    st.stop()

PDF_PATH = trouver_pdf(str(CONFIG.get("source", "")))

barre_laterale(CONFIG)

PAGES_FN = {
    "assistant": lambda: page_assistant(CONFIG),
    "document": page_document,
    "parcours": page_checklist,
    "dashboard": page_dashboard,
    "about": page_about,
}
PAGES_FN.get(st.session_state["page"], PAGES_FN["assistant"])()