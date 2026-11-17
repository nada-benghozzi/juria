"""
app.py : interface de chat Streamlit de JurIA.

Branchée sur le contrat de rag.py :
    rag.load_index(path)  -> charge l'index
    rag.ask(question)     -> {"answer", "answerable", "sources", "images", "tables"}

Lancement :
    streamlit run app.py

Prérequis :
    - le dossier d'index (index.faiss, index.pkl, originals.json, config.json)
      copié en local dans index/ (ou chemin donné par JURIA_INDEX_DIR)
    - un fichier .env avec GOOGLE_API_KEY
"""

import base64
import os

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

INDEX_DIR = os.environ.get("JURIA_INDEX_DIR", "index/")

EXEMPLES = [
    "Quelles sont les formalités pour créer une société en Tunisie ?",
    "Quels documents faut-il pour le dépôt définitif au registre du commerce ?",
    "Quelles sont les étapes pour publier au JORT ?",
]

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="JurIA", page_icon="⚖️", layout="centered")

# Style minimal pour afficher proprement les tableaux HTML extraits du PDF
st.markdown(
    """
    <style>
    .juria-table { overflow-x: auto; margin: 0.5rem 0 1rem 0; }
    .juria-table table { border-collapse: collapse; font-size: 0.85rem; width: 100%; }
    .juria-table th, .juria-table td { border: 1px solid #8884; padding: 4px 8px;
                                        text-align: left; vertical-align: top; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Chargement de l'index (une seule fois par session serveur)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement de la base juridique...")
def charger_index(path):
    """Charge l'index FAISS et renvoie sa configuration (source, comptes...)."""
    return rag.load_index(path)


# ---------------------------------------------------------------------------
# Affichage d'un résultat de ask()
# ---------------------------------------------------------------------------
def afficher_resultat(res):
    """Affiche la réponse, puis tableaux, figures et sources."""
    if not res.get("answerable", True):
        st.warning(res["answer"])
        return

    st.markdown(res["answer"])

    # Tableaux utilisés pour la réponse (HTML issu de l'extraction)
    if res.get("tables"):
        with st.expander(f"📊 Tableaux ({len(res['tables'])})", expanded=True):
            for html in res["tables"]:
                st.markdown(f'<div class="juria-table">{html}</div>',
                            unsafe_allow_html=True)

    # Figures utilisées pour la réponse (PNG en base64)
    if res.get("images"):
        with st.expander(f"🖼️ Figures ({len(res['images'])})", expanded=True):
            for img_b64 in res["images"]:
                st.image(base64.b64decode(img_b64), use_container_width=True)

    # Sources : les citées d'abord, puis les autres
    if res.get("sources"):
        with st.expander(f"📚 Sources ({len(res['sources'])})"):
            for n, s in enumerate(res["sources"], start=1):
                pages = ", ".join(map(str, s["pages"])) or "?"
                marque = "✅ citée" if s["cited"] else "consultée"
                st.markdown(
                    f"**[{n}]** {s['type']} · page(s) {pages} · "
                    f"similarité {s['score']} · {marque}"
                )
                st.caption(s["extrait"])


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ JurIA")
    st.write("Assistant d'information juridique pour les startups et PME tunisiennes.")
    st.info(
        "JurIA fournit de l'information juridique, pas un conseil. "
        "Pour un enjeu important, consultez un avocat."
    )
    st.subheader("Exemples de questions")
    for i, exemple in enumerate(EXEMPLES):
        if st.button(exemple, key=f"ex{i}", use_container_width=True):
            st.session_state["question_en_attente"] = exemple
    st.divider()
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Corps de la page
# ---------------------------------------------------------------------------
st.title("JurIA")
st.caption("Posez votre question juridique en langage courant. "
           "Les réponses s'appuient sur la base documentaire et citent leurs sources.")

# Chargement de l'index avec message d'erreur clair si absent
try:
    config = charger_index(INDEX_DIR)
except FileNotFoundError:
    st.error(
        f"Index introuvable dans « {INDEX_DIR} ». Copiez-y les fichiers produits par "
        "ingest.py (index.faiss, index.pkl, originals.json, config.json)."
    )
    st.stop()

st.sidebar.caption(f"Base : {config.get('source', '?')} · "
                   f"{config.get('indexed', '?')} éléments indexés")

# Historique de la conversation (liste de {"role", "content" ou "result"})
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            afficher_resultat(msg["result"])

# Question saisie ou choisie dans les exemples
question = st.chat_input("Votre question juridique...")
if not question:
    question = st.session_state.pop("question_en_attente", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base juridique..."):
            resultat = rag.ask(question)
        afficher_resultat(resultat)
    st.session_state["messages"].append({"role": "assistant", "result": resultat})