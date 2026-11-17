"""
app.py : interface Streamlit de JurIA (assistant juridique pour startups et PME tunisiennes).

Pages :
    💬 Assistant juridique     chat avec réponses sourcées, tableaux lisibles, figures, feedback
    📄 Analyser un document    PDF (texte) ou photo analysé avec la base juridique
    ✅ Parcours création       checklist de création de société, avec aide de l'IA à chaque étape
    📊 Tableau de bord         usage anonymisé : questions fréquentes, taux de réponse, satisfaction
    ℹ️ Offres & feuille de route   modèle d'abonnement, évolutions, responsabilité

Contrat avec le backend (rag.py) :
    rag.load_index(path)                       -> config de l'index
    rag.ask(question)                          -> dict (answer, answerable, sources, images, tables)
    rag.ask_document(question, data, mime)     -> même format

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
import time
from datetime import datetime
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

PAGE_ASSISTANT = "💬 Assistant juridique"
PAGE_DOCUMENT = "📄 Analyser un document"
PAGE_CHECKLIST = "✅ Parcours création"
PAGE_DASHBOARD = "📊 Tableau de bord"
PAGE_ABOUT = "ℹ️ Offres & feuille de route"
PAGES = [PAGE_ASSISTANT, PAGE_DOCUMENT, PAGE_CHECKLIST, PAGE_DASHBOARD, PAGE_ABOUT]

# Thématiques proposées à l'utilisateur (questions d'exemple + mots-clés pour le suivi d'usage).
# À ajuster selon ce que contient réellement la base : retirez les questions qui
# renvoient « information introuvable » pendant vos tests.
TOPICS = {
    "Création de société": {
        "icon": "🏢", "desc": "Formalités, statuts, registre du commerce",
        "keywords": ["créer", "création", "société", "statuts", "registre du commerce",
                     "jort", "immatricul", "greffe", "formalité"],
        "questions": ["Quelles sont les formalités pour créer une société en Tunisie ?",
                      "Quels documents faut-il pour le dépôt définitif au registre du commerce ?"],
    },
    "Forme juridique": {
        "icon": "⚖️", "desc": "SARL, SA, capital, associés",
        "keywords": ["sarl", "société anonyme", "forme juridique", "gérant", "capital",
                     "associé", "actionnaire"],
        "questions": ["Quelle forme juridique choisir pour ma startup ?",
                      "Quelle est la différence entre une SARL et une SA ?"],
    },
    "Fiscalité": {
        "icon": "💰", "desc": "Impôts, TVA, identifiant fiscal",
        "keywords": ["impôt", "fiscal", "tva", "taxe", "identifiant fiscal", "déclaration d'existence"],
        "questions": ["Quelles sont les obligations fiscales d'une nouvelle société ?",
                      "Comment obtenir l'identifiant fiscal de ma société ?"],
    },
    "Financement & incitations": {
        "icon": "🚀", "desc": "Aides, Startup Act, investisseurs",
        "keywords": ["financement", "startup act", "label", "incitation", "subvention",
                     "investisseur", "apii", "aide"],
        "questions": ["Quelles aides existent pour financer mon projet ?",
                      "Comment fonctionne le label Startup Act ?"],
    },
    "Recrutement": {
        "icon": "👥", "desc": "Contrat de travail, CNSS, premier salarié",
        "keywords": ["recrut", "salarié", "contrat de travail", "cnss", "employé", "embauche"],
        "questions": ["Quelles sont les obligations pour recruter un premier salarié ?",
                      "Quelles démarches faire auprès de la CNSS ?"],
    },
    "Contrats": {
        "icon": "📝", "desc": "Clauses, clients, partenaires",
        "keywords": ["contrat", "clause", "client", "fournisseur", "partenaire"],
        "questions": ["Quelles clauses vérifier dans un contrat commercial ?",
                      "Comment sécuriser un contrat avec un partenaire ?"],
    },
}

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

# Largeur pleine : « width="stretch" » dans les versions récentes de Streamlit,
# « use_container_width=True » dans les plus anciennes.
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})

TYPES_FR = {"text": ("📄", "Texte"), "table": ("📊", "Tableau"), "image": ("🖼️", "Figure")}

# ---------------------------------------------------------------------------
# Configuration de la page et style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="JurIA · Assistant juridique", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.block-container { max-width: 1000px; padding-top: 1.5rem; }
#MainMenu, footer { visibility: hidden; }

/* --- Barre latérale --- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%); }
[data-testid="stSidebar"] * { color: #E0E7FF; }
[data-testid="stSidebar"] button, [data-testid="stSidebar"] button * { color: #1E1B4B !important; }
[data-testid="stSidebar"] button { background: #EEF2FF; border: 0; border-radius: 10px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .55rem .8rem; border-radius: 10px; width: 100%; cursor: pointer; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.10); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,.20); font-weight: 600; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
.jr-brand { font-size: 1.7rem; font-weight: 800; letter-spacing: -.5px; }
.jr-brand-sub { font-size: .85rem; opacity: .8; margin-bottom: 1rem; line-height: 1.3; }
.jr-side-note { background: rgba(255,255,255,.10); border-radius: 12px; padding: .7rem .9rem;
    font-size: .8rem; line-height: 1.35; margin-top: .6rem; }

/* --- Bandeau d'en-tête --- */
.jr-hero { background: linear-gradient(135deg, #4338CA 0%, #7C3AED 60%, #C026D3 100%);
    color: #fff; border-radius: 20px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(67,56,202,.25); }
.jr-hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .2rem 0; padding: 0; }
.jr-hero p { color: #E0E7FF; margin: 0 0 .8rem 0; font-size: 1.02rem; }
.jr-hero span { display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
    padding: .25rem .8rem; margin: 0 .4rem .3rem 0; font-size: .82rem; }
.jr-title { font-size: 1.5rem; font-weight: 700; margin: 0 0 .2rem 0; color: #1E1B4B; }
.jr-subtitle { color: #5B6086; margin-bottom: 1rem; }

/* --- Messages --- */
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #E5E7F3; border-radius: 16px;
    padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(30,27,75,.05); }
.jr-cite { display: inline-flex; align-items: center; justify-content: center; min-width: 1.25rem;
    height: 1.25rem; padding: 0 .3rem; border-radius: 999px; background: #EEF2FF; color: #4338CA;
    font-size: .72rem; font-weight: 700; margin: 0 .1rem; vertical-align: text-top; }
.jr-meta { margin: .6rem 0 .4rem 0; }
.jr-chip { display: inline-block; background: #F3F4FB; color: #4B5280; border-radius: 999px;
    padding: .15rem .65rem; margin: 0 .35rem .3rem 0; font-size: .78rem; }
.jr-chip.high { background: #DCFCE7; color: #166534; }
.jr-chip.mid { background: #FEF3C7; color: #92400E; }
.jr-chip.low { background: #FEE2E2; color: #991B1B; }
.jr-callout { background: #FFF8E6; border-left: 4px solid #F5B544; border-radius: 8px;
    padding: .7rem 1rem; margin: .8rem 0; font-size: .92rem; color: #5B4510; }
.jr-empty { background: #F8F7FF; border: 1px dashed #A5B4FC; border-radius: 14px;
    padding: 1rem 1.2rem; color: #312E81; line-height: 1.5; }

/* --- Tableaux --- */
.jr-table { overflow: auto; max-height: 420px; border: 1px solid #E5E7F3; border-radius: 12px; }
.jr-table table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .85rem; }
.jr-table th { position: sticky; top: 0; background: #EEF2FF; color: #312E81; text-align: left;
    padding: .55rem .7rem; border-bottom: 2px solid #C7D2FE; font-weight: 700; }
.jr-table td { padding: .5rem .7rem; border-bottom: 1px solid #EEF0F8; vertical-align: top;
    min-width: 90px; line-height: 1.4; }
.jr-table tbody tr:nth-child(even) td { background: #FAFAFF; }
.jr-table tbody tr:hover td { background: #F1F3FF; }

/* --- Sources --- */
.jr-source { border: 1px solid #E5E7F3; border-radius: 12px; padding: .7rem .9rem; margin-bottom: .6rem; }
.jr-source-head { font-size: .88rem; margin-bottom: .35rem; }
.jr-badge { float: right; font-size: .72rem; background: #F3F4FB; color: #4B5280;
    border-radius: 999px; padding: .1rem .55rem; }
.jr-badge.ok { background: #DCFCE7; color: #166534; }
.jr-bar { height: 5px; background: #EEF0F8; border-radius: 99px; overflow: hidden; margin-bottom: .45rem; }
.jr-bar > div { height: 100%; background: linear-gradient(90deg, #6366F1, #A855F7); }
.jr-extrait { font-size: .8rem; color: #5B6086; line-height: 1.4; }

/* --- Offres --- */
.jr-plan-name { font-weight: 700; color: #4338CA; letter-spacing: .5px; font-size: .85rem; }
.jr-plan-price { font-size: 2rem; font-weight: 800; color: #1E1B4B; line-height: 1.1; }
.jr-plan-price small { font-size: .85rem; font-weight: 500; color: #5B6086; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


# ---------------------------------------------------------------------------
# Rappels (callbacks) : exécutés avant le rafraîchissement de la page
# ---------------------------------------------------------------------------
def poser_question(q):
    """Programme une question et bascule sur la page de l'assistant."""
    st.session_state["pending"] = q
    st.session_state["page"] = PAGE_ASSISTANT


def nouvelle_conversation():
    st.session_state["messages"] = []
    for k in [k for k in st.session_state if k.startswith("fb_")]:
        del st.session_state[k]


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
    """Transforme les citations [1], [2, 3] en pastilles colorées."""
    texte = texte.replace("<", "&lt;")

    def remplacer(m):
        nums = re.split(r"\s*,\s*", m.group(1))
        return "".join(f'<span class="jr-cite">{n}</span>' for n in nums)

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", remplacer, texte)


class _TableParser(HTMLParser):
    """Lit un tableau HTML et en extrait les lignes de texte."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_JUNK = re.compile(r"^[\W_]+$")  # cellule ne contenant que des symboles (« | », « ! »...)


def tableau_vers_lignes(source_html):
    """Extrait un tableau propre : supprime cellules parasites OCR, lignes et colonnes vides."""
    parser = _TableParser()
    try:
        parser.feed(source_html)
    except Exception:
        return []
    rows = parser.rows
    if not rows:
        return []
    largeur = max(len(r) for r in rows)
    rows = [r + [""] * (largeur - len(r)) for r in rows]
    rows = [["" if _JUNK.match(c) else c for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    garder = [j for j in range(largeur) if any(r[j] for r in rows)]
    return [[r[j] for j in garder] for r in rows]


def tableau_html(rows):
    entete, *corps = rows
    th = "".join(f"<th>{html.escape(c)}</th>" for c in entete)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                  for r in corps)
    return (f'<div class="jr-table"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{trs}</tbody></table></div>")


def afficher_tableau(source_html, cle):
    """Affiche un tableau lisible (en-tête fixe, lignes alternées) + export CSV."""
    rows = tableau_vers_lignes(source_html)
    if len(rows) < 1 or sum(len(r) for r in rows) < 2:
        # structure inexploitable : on montre simplement le texte brut
        st.caption(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", source_html)).strip()[:1500])
        return
    st.markdown(tableau_html(rows), unsafe_allow_html=True)
    tampon = io.StringIO()
    csv.writer(tampon).writerows(rows)
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{max(len(rows) - 1, 0)} ligne(s) · {len(rows[0])} colonne(s). "
               "Extrait d'un document PDF : des erreurs de lecture (OCR) sont possibles.")
    c2.download_button("⬇️ CSV", data=tampon.getvalue().encode("utf-8-sig"),
                       file_name="tableau_juria.csv", mime="text/csv", key=f"csv_{cle}")


def afficher_image(octets_b64, legende):
    donnees = base64.b64decode(octets_b64)
    for options in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(donnees, caption=legende, **options)
            return
        except Exception:  # selon la version de Streamlit, le paramètre diffère
            continue


def nettoyer_extrait(texte):
    """Retire les symboles isolés issus de l'OCR pour un extrait plus lisible."""
    return re.sub(r"\s+", " ", re.sub(r"(?:(?<=\s)|^)[|!;'’‘_]+(?=\s|$)", " ", texte)).strip()


def afficher_sources(res):
    sources = res.get("sources", [])
    if not sources:
        return
    ordre = sorted(enumerate(sources, start=1), key=lambda t: (not t[1]["cited"], -t[1]["score"]))
    with st.expander(f"📚 Sources consultées ({len(sources)})"):
        for n, s in ordre:
            icone, nom = TYPES_FR.get(s["type"], ("📄", "Texte"))
            pages = ", ".join(map(str, s["pages"])) or "?"
            badge = ('<span class="jr-badge ok">Citée</span>' if s["cited"]
                     else '<span class="jr-badge">Consultée</span>')
            largeur = max(0, min(100, int(s["score"] * 100)))
            st.markdown(
                f'<div class="jr-source"><div class="jr-source-head">'
                f'<span class="jr-cite">{n}</span> {icone} <b>{nom}</b> · page {pages}{badge}</div>'
                f'<div class="jr-bar"><div style="width:{largeur}%"></div></div>'
                f'<div class="jr-extrait">{html.escape(nettoyer_extrait(s["extrait"]))}</div></div>',
                unsafe_allow_html=True)
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
    """Effet « machine à écrire » (la réponse est déjà complète, c'est purement visuel)."""
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
        f'<div class="jr-empty"><b>🔍 Je n\'ai pas trouvé cette information</b><br>{html.escape(detail)}'
        "<br>Je préfère ne pas répondre au hasard. 💡 Essayez de reformuler avec des termes plus "
        "précis, ou consultez un avocat pour ce point.</div>", unsafe_allow_html=True)
    if question:
        st.caption("Pour continuer, vous pouvez essayer :")
        for i, q in enumerate(suggestions_liees(question, 2)):
            st.button(q, key=f"alt_{cle}_{i}", on_click=poser_question, args=(q,),
                      **STRETCH)


def afficher_reponse(res, cle, idx=None, derniere=False, question="", animer=False):
    """Affiche un résultat de rag.ask() : réponse, alerte, tableaux, figures, sources, feedback."""
    if res.get("error"):
        st.error(res["answer"])
        st.caption("Vérifiez la clé API et la connexion, puis réessayez.")
        if question:
            st.button("🔄 Réessayer", key=f"retry_{cle}", on_click=poser_question, args=(question,))
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

    # Pastilles d'information
    chips = []
    pert = niveau_pertinence(res)
    if pert:
        chips.append(f'<span class="jr-chip {pert[1]}">{pert[0]}</span>')
    if res.get("sources"):
        chips.append(f'<span class="jr-chip">📚 {len(res["sources"])} sources</span>')
    if res.get("tables"):
        chips.append(f'<span class="jr-chip">📊 {len(res["tables"])} tableau(x)</span>')
    if res.get("images"):
        chips.append(f'<span class="jr-chip">🖼️ {len(res["images"])} figure(s)</span>')
    if res.get("latency") is not None:
        chips.append(f'<span class="jr-chip">⏱ {str(res["latency"]).replace(".", ",")} s</span>')
    if chips:
        st.markdown(f'<div class="jr-meta">{"".join(chips)}</div>', unsafe_allow_html=True)

    # Alerte « enjeu élevé » si l'IA recommande un avocat
    if re.search(r"avocat", texte, re.IGNORECASE):
        st.markdown('<div class="jr-callout">⚖️ <b>Enjeu important.</b> JurIA donne de l\'information '
                    "juridique, pas un conseil : pour sécuriser votre décision, faites-la valider par un "
                    "avocat (un avis d'avocat en option payante est prévu, voir « Offres »).</div>",
                    unsafe_allow_html=True)

    # Tableaux, figures et sources
    if res.get("tables") and st.session_state.get("show_tables", True):
        for i, t in enumerate(res["tables"]):
            with st.expander(f"📊 Tableau {i + 1}", expanded=(i == 0)):
                afficher_tableau(t, f"{cle}_{i}")
    if res.get("images") and st.session_state.get("show_images", True):
        with st.expander(f"🖼️ Figures ({len(res['images'])})", expanded=True):
            for i, img in enumerate(res["images"]):
                afficher_image(img, f"Figure {i + 1}")
    if st.session_state.get("show_sources", True):
        afficher_sources(res)

    # Feedback et pistes pour poursuivre
    if idx is not None and hasattr(st, "feedback"):
        st.caption("Cette réponse vous a-t-elle été utile ?")
        st.feedback("thumbs", key=f"fb_{idx}", on_change=enregistrer_feedback, args=(idx,))
    if derniere and question:
        st.caption("💡 Pour aller plus loin :")
        for i, q in enumerate(suggestions_liees(question)):
            st.button(q, key=f"sug_{cle}_{i}", on_click=poser_question, args=(q,),
                      **STRETCH)


# ---------------------------------------------------------------------------
# Page 1 : assistant
# ---------------------------------------------------------------------------
def afficher_accueil(config):
    counts = config.get("counts", {})
    st.markdown(
        '<div class="jr-hero"><h1>⚖️ JurIA</h1>'
        "<p>Posez vos questions juridiques en langage courant. Réponses claires, sourcées et "
        "adaptées au contexte tunisien.</p>"
        "<span>🔎 Réponses citant leurs sources</span><span>📊 Tableaux et figures</span>"
        "<span>📄 Analyse de vos documents</span><span>🇹🇳 Startups et PME tunisiennes</span></div>",
        unsafe_allow_html=True)
    if counts:
        c1, c2, c3 = st.columns(3)
        c1.metric("Passages de texte", counts.get("text", 0))
        c2.metric("Tableaux", counts.get("table", 0))
        c3.metric("Figures", counts.get("image", 0))
    st.markdown('<div class="jr-title">Par où commencer ?</div>'
                '<div class="jr-subtitle">Choisissez une thématique ou écrivez votre question ci-dessous.</div>',
                unsafe_allow_html=True)
    noms = list(TOPICS)
    for ligne in range(0, len(noms), 3):
        cols = st.columns(3)
        for col, nom in zip(cols, noms[ligne:ligne + 3]):
            t = TOPICS[nom]
            with col.container(border=True):
                st.markdown(f"**{t['icon']} {nom}**")
                st.caption(t["desc"])
                for i, q in enumerate(t["questions"]):
                    st.button(q, key=f"home_{nom}_{i}", on_click=poser_question, args=(q,),
                              **STRETCH)


def traiter_question(question):
    """Pose la question à JurIA et affiche la réponse dans la conversation."""
    messages = st.session_state["messages"]
    question = question.strip()[:MAX_QUESTION_CHARS]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("JurIA analyse votre question…", expanded=False) as status:
            st.write("🔎 Recherche des passages pertinents dans la base juridique")
            res, depuis_cache = interroger(question)
            st.write("✍️ Rédaction de la réponse")
            status.update(label="Réponse prête", state="complete")
        if depuis_cache:
            res["latency"] = 0.0
        idx = len(messages)
        theme = detecter_thematique(question)
        messages.append({"role": "assistant", "result": res, "question": question, "topic": theme})
        if not res.get("error"):
            log_event({"type": "question", "question": question, "topic": theme,
                       "answerable": bool(res.get("answerable")),
                       "latency": res.get("latency") if not depuis_cache else None})
        afficher_reponse(res, f"m{idx}", idx=idx, derniere=True, question=question, animer=True)


def page_assistant(config):
    messages = st.session_state["messages"]
    if not messages:
        afficher_accueil(config)
    for i, m in enumerate(messages):
        if m["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                afficher_reponse(m["result"], f"m{i}", idx=i, derniere=(i == len(messages) - 1),
                                 question=m["question"])
    question = st.chat_input("Posez votre question juridique…", max_chars=MAX_QUESTION_CHARS)
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        traiter_question(question)


# ---------------------------------------------------------------------------
# Page 2 : analyse de document
# ---------------------------------------------------------------------------
def page_document():
    st.markdown('<div class="jr-title">📄 Analyser un document</div>'
                '<div class="jr-subtitle">Déposez un contrat, un formulaire ou un courrier : JurIA le lit '
                "et le confronte à sa base juridique.</div>", unsafe_allow_html=True)
    fichier = st.file_uploader("Document (PDF avec texte, ou photo PNG/JPG)",
                               type=["pdf", "png", "jpg", "jpeg"])
    question = st.text_area("Que voulez-vous savoir ? (facultatif)", height=90, max_chars=MAX_QUESTION_CHARS,
                            placeholder="Ex. : Quelles clauses sont risquées ? Ce document est-il complet ?")
    trop_gros = bool(fichier) and fichier.size > MAX_UPLOAD_MB * 1024 * 1024
    if trop_gros:
        st.error(f"Fichier trop volumineux (maximum {MAX_UPLOAD_MB} Mo).")
    if fichier and fichier.type.startswith("image/"):
        st.image(fichier.getvalue(), caption=fichier.name, width=320)

    if st.button("🔍 Analyser le document", type="primary", disabled=(fichier is None or trop_gros)):
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
# Page 3 : parcours de création
# ---------------------------------------------------------------------------
def page_checklist():
    st.markdown('<div class="jr-title">✅ Parcours de création de société</div>'
                '<div class="jr-subtitle">Suivez les étapes une par une et demandez à JurIA le détail '
                "des documents de chacune.</div>", unsafe_allow_html=True)
    faites = sum(1 for i in range(len(CHECKLIST)) if st.session_state.get(f"chk_{i}"))
    st.progress(faites / len(CHECKLIST), text=f"{faites} étape(s) sur {len(CHECKLIST)} terminée(s)")
    for i, (etape, organisme, question) in enumerate(CHECKLIST):
        with st.container(border=True):
            c1, c2, c3 = st.columns([0.7, 5, 2.6], vertical_alignment="center")
            c1.checkbox(f"Étape {i + 1}", key=f"chk_{i}", label_visibility="collapsed")
            c2.markdown(f"**{i + 1}. {etape}**  \n:gray[Auprès de : {organisme}]")
            c3.button("💬 Détails avec JurIA", key=f"chk_btn_{i}", on_click=poser_question,
                      args=(question,), **STRETCH)
    if faites == len(CHECKLIST):
        st.success("🎉 Toutes les étapes sont cochées. Vérifiez auprès des administrations que votre "
                   "dossier est complet.")
    st.caption("Étapes issues du guide indexé (formalités de création des sociétés). La coche est un "
               "aide-mémoire local : elle n'est pas enregistrée.")


# ---------------------------------------------------------------------------
# Page 4 : tableau de bord
# ---------------------------------------------------------------------------
def page_dashboard():
    st.markdown('<div class="jr-title">📊 Tableau de bord</div>'
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
    ups = sum(1 for f in feedbacks if f.get("value") == "up")
    downs = sum(1 for f in feedbacks if f.get("value") == "down")
    latences = pd.to_numeric(df.get("latency"), errors="coerce").dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions posées", len(df))
    c2.metric("Taux de réponse", f"{round(100 * df['answerable'].mean())} %")
    c3.metric("Temps moyen", f"{latences.mean():.1f} s" if len(latences) else "n/a")
    c4.metric("Satisfaction", f"{round(100 * ups / (ups + downs))} %" if ups + downs else "n/a")

    st.markdown("##### Thématiques les plus demandées")
    st.bar_chart(df["topic"].value_counts(), color="#4338CA")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Questions les plus fréquentes")
        st.dataframe(df["question"].value_counts().head(10).rename("Fois"), **STRETCH)
    with g2:
        st.markdown("##### Questions sans réponse")
        sans = df[~df["answerable"]]["question"].drop_duplicates().tolist()
        if sans:
            for q in sans[:10]:
                st.markdown(f"- {q}")
            st.caption("À examiner : ce sont les sujets à ajouter à la base documentaire.")
        else:
            st.success("Toutes les questions ont reçu une réponse.")
    st.download_button("⬇️ Exporter le journal (CSV)", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="usage_juria.csv", mime="text/csv")
    st.caption("Seules la question, la thématique et le résultat sont enregistrés, sans identifiant "
               "d'utilisateur.")


# ---------------------------------------------------------------------------
# Page 5 : offres, feuille de route, responsabilité
# ---------------------------------------------------------------------------
def page_about():
    st.markdown('<div class="jr-title">ℹ️ À propos de JurIA</div>'
                '<div class="jr-subtitle">Modèle économique, évolutions prévues et engagements.</div>',
                unsafe_allow_html=True)
    t_offres, t_route, t_resp = st.tabs(["💎 Offres", "🗺️ Feuille de route", "🛡️ Responsabilité"])

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
                    st.markdown(f"✓ {a}")
        st.caption("Organisations d'appui (incubateurs, programmes) : licences annuelles et tableau de bord "
                   "des questions les plus posées par cohorte.")

    with t_route:
        etapes = [
            ("Enrichissement juridique", "Prototype", "Base actuelle : un guide de l'entrepreneur. "
             "Extension prévue : fiscal, social, commercial, propriété intellectuelle."),
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
        risques = pd.DataFrame({
            "Risque": ["Hallucinations de l'IA", "Textes obsolètes", "Fuite de données",
                       "Mauvaise interprétation juridique", "Biais de l'IA"],
            "Mesure de prévention": ["RAG + sources vérifiables", "Mise à jour du corpus",
                                     "Chiffrement et contrôle d'accès", "Orientation vers un avocat",
                                     "Tests et contrôle des réponses"],
        })
        st.dataframe(risques, hide_index=True, **STRETCH)


# ---------------------------------------------------------------------------
# Barre latérale et exports
# ---------------------------------------------------------------------------
def exporter_conversation():
    """Conversation au format Markdown (pour archivage ou partage)."""
    lignes = ["# Conversation JurIA", f"_Exportée le {datetime.now():%d/%m/%Y à %H:%M}_", ""]
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            lignes += [f"## ❓ {m['content']}", ""]
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
    with st.sidebar:
        st.markdown('<div class="jr-brand">⚖️ JurIA</div><div class="jr-brand-sub">Assistant juridique '
                    "intelligent<br>pour startups et PME tunisiennes</div>", unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        with st.expander("⚙️ Affichage"):
            st.toggle("Tableaux", value=True, key="show_tables")
            st.toggle("Figures", value=True, key="show_images")
            st.toggle("Sources", value=True, key="show_sources")
            st.toggle("Effet de frappe", value=True, key="animation")
        if st.session_state["messages"]:
            st.download_button("⬇️ Exporter la conversation", data=exporter_conversation(),
                               file_name="conversation_juria.md", mime="text/markdown",
                               **STRETCH)
            st.button("🗑️ Nouvelle conversation", on_click=nouvelle_conversation,
                      **STRETCH)
        st.markdown('<div class="jr-side-note">⚠️ JurIA fournit de l\'information juridique, pas un '
                    "conseil. Pour un enjeu important, consultez un avocat.</div>", unsafe_allow_html=True)
        st.caption(f"Base : {config.get('source', '?')} · {config.get('indexed', '?')} éléments")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("page", PAGE_ASSISTANT)

try:
    CONFIG = charger_index(INDEX_DIR)
except FileNotFoundError:
    st.error(f"Index introuvable dans « {INDEX_DIR} ». Copiez-y les fichiers produits par ingest.py "
             "(index.faiss, index.pkl, originals.json, config.json).")
    st.stop()

barre_laterale(CONFIG)

page = st.session_state["page"]
if page == PAGE_ASSISTANT:
    page_assistant(CONFIG)
elif page == PAGE_DOCUMENT:
    page_document()
elif page == PAGE_CHECKLIST:
    page_checklist()
elif page == PAGE_DASHBOARD:
    page_dashboard()
else:
    page_about()