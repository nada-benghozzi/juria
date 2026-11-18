"""
explore_index.py : catalogue du contenu réel de l'index JurIA.

Objectif : choisir des questions de démo qui s'appuient sur des tableaux et des
figures qui existent vraiment et qui s'affichent bien.

Produit :
    tests/catalogue.md        liste des textes / tableaux / figures (page, résumé, qualité)
    tests/figures/*.png       (option --export-images) figures à regarder avant de choisir
    tests/candidates.json     (option --suggest N) questions candidates générées par l'IA

Usage (à lancer depuis la racine du dépôt) :
    python tools/explore_index.py
    python tools/explore_index.py --export-images
    python tools/explore_index.py --suggest 8
"""

import argparse
import base64
import json
import os
import re
import sys
from html.parser import HTMLParser

SEUIL_QUALITE = 0.78   # même seuil que app.py


# ---------------------------------------------------------------------------
# Qualité d'un tableau (version simplifiée de celle d'app.py)
# ---------------------------------------------------------------------------
class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

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


MOTS_COURTS = {"de", "du", "la", "le", "les", "un", "une", "au", "aux", "en", "et", "ou", "à", "a",
               "d", "l", "par", "sur", "si", "ne", "pas", "se", "sa", "ses", "son", "ce", "n°"}


def analyser_tableau(html_src):
    """Retourne (nb_lignes, nb_colonnes, score_ocr, verdict)."""
    p = _TableParser()
    try:
        p.feed(html_src)
    except Exception:
        return 0, 0, 0.0, "illisible"
    rows = [r for r in p.rows if any(r)]
    if not rows:
        return 0, 0, 0.0, "illisible"
    largeur = max(len(r) for r in rows)
    cellules = [c for r in rows for c in r + [""] * (largeur - len(r))]
    remplies = [c for c in cellules if c.strip()]
    taux = len(remplies) / max(len(cellules), 1)
    mots_par_cellule = sum(len(c.split()) for c in remplies) / max(len(remplies), 1)

    mots = [m for c in remplies for m in c.split()]
    mauvais = 0
    for m in mots:
        coeur = m.strip(".,;:()«»\"'’-")
        if not coeur or re.search(r"[|!_*~^]", coeur):
            mauvais += 1
        elif len(coeur) <= 2 and coeur.lower() not in MOTS_COURTS and not coeur.isdigit():
            mauvais += 1
    score = 1 - mauvais / max(len(mots), 1)

    if len(cellules) >= 6 and ((taux < 0.5 and mots_par_cellule < 3) or
                               (len(rows) >= 3 and mots_par_cellule < 1.6)):
        verdict = "structure cassée"
    elif score < SEUIL_QUALITE:
        verdict = "OCR dégradé"
    else:
        verdict = "bon"
    return len(rows), largeur, round(score, 2), verdict


def texte_brut(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
def charger(index_dir):
    with open(os.path.join(index_dir, "originals.json"), encoding="utf-8") as f:
        originals = json.load(f)
    items = []
    for doc_id, it in originals.items():
        it = dict(it)
        it["id"] = doc_id[:8]
        items.append(it)
    items.sort(key=lambda x: (x["type"], x["pages"][0] if x["pages"] else 0))
    return items


def ecrire_catalogue(items, chemin):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    lignes = ["# Catalogue de l'index JurIA", ""]
    for kind, titre in (("table", "Tableaux"), ("image", "Figures"), ("text", "Textes")):
        groupe = [i for i in items if i["type"] == kind]
        lignes += [f"## {titre} ({len(groupe)})", ""]
        for it in groupe:
            pages = ", ".join(map(str, it["pages"])) or "?"
            if kind == "table":
                nl, nc, score, verdict = analyser_tableau(it["content"])
                tete = f"- **p.{pages}** [{it['id']}] {nl} lignes x {nc} col. | OCR {score} | **{verdict}**"
            else:
                tete = f"- **p.{pages}** [{it['id']}]"
            lignes.append(tete)
            lignes.append(f"  - {it['summary'][:220]}")
        lignes.append("")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    print(f"Catalogue écrit : {chemin}")


def exporter_images(items, dossier):
    os.makedirs(dossier, exist_ok=True)
    n = 0
    for it in items:
        if it["type"] != "image":
            continue
        pages = "-".join(map(str, it["pages"])) or "x"
        chemin = os.path.join(dossier, f"p{pages}_{it['id']}.png")
        with open(chemin, "wb") as f:
            f.write(base64.b64decode(it["content"]))
        n += 1
    print(f"{n} figures exportées dans {dossier} (regardez-les avant de choisir)")


# ---------------------------------------------------------------------------
# Suggestion de questions (IA) : une question par élément choisi
# ---------------------------------------------------------------------------
PROMPT_QUESTION = (
    "You help build a demo for a legal assistant for Tunisian startups. Below is the "
    "content of one {kind} extracted from a Tunisian legal guide. Write ONE question in "
    "French that an entrepreneur could ask, that can be answered ONLY from this content, "
    "and that does not mention 'the table', 'the figure' or 'the text'. "
    "Return only the question.\n\n{content}"
)


def suggerer(items, n, chemin):
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=os.environ.get("JURIA_GEN_MODEL", "gemini-3.1-flash-lite"),
                                 temperature=0.3)

    choix = []
    # Tableaux lisibles uniquement
    bons = [i for i in items if i["type"] == "table"
            and analyser_tableau(i["content"])[3] == "bon" and analyser_tableau(i["content"])[0] >= 3]
    choix += [("table", i) for i in bons[:n]]
    # Figures : on part du résumé de la figure
    choix += [("image", i) for i in [i for i in items if i["type"] == "image"][:n]]
    # Textes assez riches
    riches = [i for i in items if i["type"] == "text" and len(i["content"]) > 800]
    choix += [("text", i) for i in riches[:n]]

    candidats = []
    for kind, it in choix:
        contenu = it["summary"] if kind == "image" else texte_brut(it["content"])[:2500]
        try:
            q = llm.invoke(PROMPT_QUESTION.format(kind=kind, content=contenu)).content
            q = (q if isinstance(q, str) else str(q)).strip().strip('"')
        except Exception as exc:
            print(f"  ! {kind} p.{it['pages']} : {type(exc).__name__}")
            continue
        candidats.append({"question": q, "expected_type": kind, "expected_pages": it["pages"]})
        print(f"  [{kind} p.{it['pages']}] {q}")

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(candidats, f, ensure_ascii=False, indent=2)
    print(f"\n{len(candidats)} questions candidates écrites : {chemin}")
    print("Relisez-les, corrigez ou ajoutez les vôtres, puis lancez tools/eval_questions.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Explore l'index JurIA.")
    ap.add_argument("--index", default="index/")
    ap.add_argument("--out", default="tests/")
    ap.add_argument("--export-images", action="store_true")
    ap.add_argument("--suggest", type=int, default=0,
                    help="génère N questions candidates par type (tableau, figure, texte)")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.index, "originals.json")):
        sys.exit(f"originals.json introuvable dans {args.index}")

    items = charger(args.index)
    ecrire_catalogue(items, os.path.join(args.out, "catalogue.md"))
    if args.export_images:
        exporter_images(items, os.path.join(args.out, "figures"))
    if args.suggest:
        suggerer(items, args.suggest, os.path.join(args.out, "candidates.json"))