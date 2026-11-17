"""
ingest.py : construction de l'index JurIA à partir d'un PDF juridique.

À lancer une seule fois (de préférence sur Google Colab, l'extraction
"hi_res" est lourde en mémoire). Produit dans le dossier de sortie :
    - index.faiss / index.pkl : index vectoriel des RÉSUMÉS
    - originals.json          : contenus ORIGINAUX (texte, tableau HTML, image base64)
    - config.json             : paramètres utiles à rag.py (modèle d'embeddings...)
    - extraction.json         : cache de l'extraction (évite de la refaire)

Usage :
    python ingest.py --pdf data/source.pdf --out index/
    python ingest.py --pdf data/source.pdf --out index/ --reuse-extraction
"""

import argparse
import base64
import hashlib
import io
import json
import os
import uuid

from dotenv import load_dotenv
from PIL import Image

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
SUMMARY_MODEL = "openai/gpt-oss-20b"        # résumés des textes et tableaux (Groq)
# Si une erreur "model not found" apparaît,
# le remplacer par un modèle Gemini disponible avec votre clé API.
VISION_MODEL = "gemini-3.1-flash-lite"      # description des images (Gemini)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_IMAGE_SIDE = 1600       # on réduit les grandes images (coût API + stockage)
SUMMARY_CONCURRENCY = 3     # appels simultanés (limiter pour éviter les 429)
VISION_CONCURRENCY = 2

TEXT_SUMMARY_PROMPT = (
    "You summarize an excerpt from a Tunisian legal document so that it can "
    "be found by a search engine. Write the summary in French, in at most 100 "
    "words. Keep legal terms, article numbers, deadlines, amounts and names "
    "of laws exactly as written. Return only the summary.\n\n"
    "Excerpt:\n{element}"
)
TABLE_SUMMARY_PROMPT = (
    "Below is a table (HTML) extracted from a legal document. Summarize it in "
    "French, in at most 100 words: purpose of the table, columns, key values "
    "and important figures. Return only the summary.\n\n"
    "Table:\n{element}"
)
IMAGE_PROMPT = (
    "Describe this diagram or figure extracted from a legal document. Write "
    "the description in French, in at most 100 words: type of figure, "
    "elements, any text you can read, relations or steps. Return only the "
    "description."
)


# ----------------------------------------------------------------------------
# 1. Extraction du PDF
# ----------------------------------------------------------------------------
def _pages(*values):
    """Retourne la liste triée des numéros de page valides (sans doublon)."""
    return sorted({v for v in values if v})


def normalize_image(image_b64, min_side):
    """Convertit l'image en PNG RGB redimensionné ; None si illisible ou trop petite."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception:
        return None
    if min(img.size) < min_side:      # logos, icônes, filets décoratifs
        return None
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_elements(pdf_path, min_image_side):
    """Découpe le PDF en blocs de texte, tableaux et images (avec leurs pages)."""
    chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",                      # détection de mise en page (tableaux, figures)
        languages=["fra", "eng"],               # OCR : français + termes anglais
        infer_table_structure=True,             # tableaux en HTML
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,    # images en base64
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts, tables, images = [], [], []
    seen_images = set()

    for chunk in chunks:
        kind = type(chunk).__name__
        meta = chunk.metadata

        if kind.startswith("Table"):            # Table ou TableChunk
            html = meta.text_as_html or chunk.text
            tables.append({"type": "table", "content": html,
                           "pages": _pages(meta.page_number)})
            continue

        # Bloc de texte (CompositeElement) : on retrouve ses pages et ses images
        originals = meta.orig_elements or []
        pages = _pages(*[el.metadata.page_number for el in originals]) \
            or _pages(meta.page_number)
        texts.append({"type": "text", "content": chunk.text, "pages": pages})

        for el in originals:
            if type(el).__name__ != "Image" or not el.metadata.image_base64:
                continue
            norm = normalize_image(el.metadata.image_base64, min_image_side)
            if norm is None:
                continue
            digest = hashlib.md5(norm.encode()).hexdigest()
            if digest in seen_images:           # image déjà vue (doublon)
                continue
            seen_images.add(digest)
            images.append({"type": "image", "content": norm,
                           "pages": _pages(el.metadata.page_number)})

    return texts + tables + images


# ----------------------------------------------------------------------------
# 2. Résumés (texte et tableaux : Groq ; images : Gemini)
# ----------------------------------------------------------------------------
def safe_batch(chain, inputs, concurrency, label):
    """Exécute chain.batch ; un échec individuel donne None au lieu de planter."""
    if not inputs:
        return []
    results = chain.batch(
        inputs,
        config={"max_concurrency": concurrency},
        return_exceptions=True,
    )
    cleaned = []
    for i, res in enumerate(results):
        if isinstance(res, Exception) or not str(res).strip():
            print(f"  ! {label} #{i} : résumé impossible ({type(res).__name__})")
            cleaned.append(None)
        else:
            cleaned.append(str(res).strip())
    return cleaned


def summarize_items(items):
    """Ajoute à chaque élément son résumé ; renvoie la liste des éléments résumés."""
    retry = dict(stop_after_attempt=5, wait_exponential_jitter=True)

    text_llm = ChatGroq(model=SUMMARY_MODEL).with_retry(**retry)
    vision_llm = ChatGoogleGenerativeAI(model=VISION_MODEL).with_retry(**retry)

    text_chain = (ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
                  | text_llm | StrOutputParser())
    table_chain = (ChatPromptTemplate.from_template(TABLE_SUMMARY_PROMPT)
                   | text_llm | StrOutputParser())
    image_chain = vision_llm | StrOutputParser()

    summarized = []
    for kind, label in [("text", "texte"), ("table", "tableau"), ("image", "image")]:
        group = [it for it in items if it["type"] == kind]
        if kind == "image":
            inputs = [[HumanMessage(content=[
                {"type": "text", "text": IMAGE_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{it['content']}"}},
            ])] for it in group]
            summaries = safe_batch(image_chain, inputs, VISION_CONCURRENCY, label)
        else:
            chain = text_chain if kind == "text" else table_chain
            inputs = [{"element": it["content"]} for it in group]
            summaries = safe_batch(chain, inputs, SUMMARY_CONCURRENCY, label)

        for it, summary in zip(group, summaries):
            if summary:
                summarized.append({**it, "summary": summary})
        print(f"  {label}s : {sum(1 for s in summaries if s)}/{len(group)} résumés réussis")

    return summarized


# ----------------------------------------------------------------------------
# 3. Construction et sauvegarde de l'index
# ----------------------------------------------------------------------------
def build_index(pdf_path, out_dir, reuse_extraction=False, min_image_side=120):
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Variable d'environnement manquante : {key}")

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "extraction.json")

    # --- Étape 1 : extraction (ou lecture du cache) ---
    if reuse_extraction and os.path.exists(cache_path):
        print("[1/4] Extraction : lecture du cache")
        with open(cache_path, encoding="utf-8") as f:
            items = json.load(f)
    else:
        print("[1/4] Extraction du PDF (hi_res, peut prendre plusieurs minutes)...")
        items = extract_elements(pdf_path, min_image_side)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
    counts = {k: sum(1 for it in items if it["type"] == k)
              for k in ("text", "table", "image")}
    print(f"      -> {counts['text']} textes, {counts['table']} tableaux, "
          f"{counts['image']} images")

    # --- Étape 2 : résumés ---
    print("[2/4] Résumés...")
    summarized = summarize_items(items)

    # --- Étape 3 : index vectoriel des résumés ---
    print("[3/4] Embeddings et index FAISS...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    source_name = os.path.basename(pdf_path)
    records, docs = {}, []
    for it in summarized:
        doc_id = str(uuid.uuid4())
        records[doc_id] = it                     # original + résumé
        docs.append(Document(
            page_content=it["summary"],
            metadata={"doc_id": doc_id, "type": it["type"],
                      "pages": it["pages"], "source": source_name},
        ))
    vectorstore = FAISS.from_documents(docs, embeddings)

    # --- Étape 4 : sauvegarde ---
    print("[4/4] Sauvegarde dans", out_dir)
    vectorstore.save_local(out_dir)              # index.faiss + index.pkl
    with open(os.path.join(out_dir, "originals.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding_model": EMBEDDING_MODEL, "source": source_name,
                   "counts": counts, "indexed": len(docs)}, f,
                  ensure_ascii=False, indent=2)

    # --- Aperçu pour vérification visuelle ---
    print("\nExemples de résumés indexés :")
    for kind in ("text", "table", "image"):
        sample = next((d for d in docs if d.metadata["type"] == kind), None)
        if sample:
            print(f"- [{kind}, pages {sample.metadata['pages']}] "
                  f"{sample.page_content[:200]}")
    print(f"\nTerminé : {len(docs)} éléments indexés.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit l'index JurIA.")
    parser.add_argument("--pdf", default="data/source.pdf")
    parser.add_argument("--out", default="index/")
    parser.add_argument("--reuse-extraction", action="store_true",
                        help="réutilise extraction.json s'il existe")
    parser.add_argument("--min-image-side", type=int, default=120,
                        help="ignore les images dont le plus petit côté est inférieur (px)")
    args = parser.parse_args()
    build_index(args.pdf, args.out, args.reuse_extraction, args.min_image_side)