#!/usr/bin/env python3
# index_jsonl_to_chroma_fixed.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, json, re, unicodedata, hashlib, signal, logging # ADDED logging
from typing import List, Dict, Iterable
from tqdm import tqdm
from unidecode import unidecode

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = "./chroma_quest"
COLLECTION_NAME = "quest_documents"
DOCUMENT_PATH = "/orcd/home/002/joycequ/quest_data/documents.jsonl"
LOG_FILE = "indexing_progress.log"

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 80
BATCH_SIZE = 256
CLEAR_COLLECTION = True
DEVICE = None

# ADDED: Set up a logger instance
logger = logging.getLogger(__name__)

def normalize_title_slug(s: str) -> str:
    if not s:
        return "untitled"
    t = unicodedata.normalize("NFC", s).strip()
    t = unidecode(t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Za-z0-9 _\-.]", "", t).strip().replace(" ", "_")
    return t or "untitled"

def stable_entity_id(title: str, text: str) -> str:
    slug = normalize_title_slug(title)
    h = hashlib.sha1((title + "\n" + (text or "")).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # CHANGED: from print to logger.warning
                logger.warning(f"Skipping malformed JSON on line {idx}: {e}")

def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chunk_by_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    if not text:
        return []
    toks = tokenizer.encode(text, add_special_tokens=False)
    if not toks:
        return []
    chunks = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(toks), step):
        end = min(start + chunk_tokens, len(toks))
        sub = toks[start:end]
        if not sub:
            break
        chunk_text = tokenizer.decode(sub, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(toks):
            break
    return chunks

class STEmbeddingFn:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 64):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        embs = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

def upsert_in_batches(collection, ids: List[str], documents: List[str], metadatas: List[Dict], batch_size: int):
    for i in range(0, len(ids), batch_size):
        j = i + batch_size
        collection.upsert(ids=ids[i:j], documents=documents[i:j], metadatas=metadatas[i:j])

def index_jsonl(jsonl_path: str):
    logger.info("Starting indexing process...")
    logger.info(f"Source file: {jsonl_path}")
    logger.info(f"Chroma DB persistence directory: {PERSIST_DIR}")
    logger.info(f"Collection name: {COLLECTION_NAME}")
    logger.info(f"Embedding model: {MODEL_NAME}")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    logger.info("ChromaDB client initialized.")

    embed_fn = STEmbeddingFn(model_name=MODEL_NAME, device=DEVICE)

    if CLEAR_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            # CHANGED: from print to logger.info
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    logger.info(f"Successfully got or created collection '{COLLECTION_NAME}'.")

    tokenizer = build_tokenizer(MODEL_NAME)

    total_docs = sum(1 for _ in read_jsonl(jsonl_path))
    docs_iter = read_jsonl(jsonl_path)
    logger.info(f"Found {total_docs} documents to process.")

    p_docs = tqdm(total=total_docs, desc="Documents processed", unit="doc")
    p_chunks = tqdm(total=0, desc="Chunks indexed", unit="chunk")

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict] = []
    running_chunk_total = 0

    interrupted = {"flag": False}
    def handle_sigint(sig, frame):
        interrupted["flag"] = True
        # CHANGED: from print to logger.warning
        logger.warning("\nInterrupt received. Flushing pending batch before exit...")

    old_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        for raw in docs_iter:
            if interrupted["flag"]:
                break

            title = (raw.get("title") or "").strip() or "untitled"
            text = (raw.get("text") or "").strip()
            entity_id = stable_entity_id(title, text)

            chunks = chunk_by_tokens(text, tokenizer, CHUNK_TOKENS, OVERLAP_TOKENS)
            if not chunks:
                chunks = [title]

            n_chunks = len(chunks)
            for idx, chunk in enumerate(chunks):
                cid = f"{entity_id}__{idx:04d}"
                batch_ids.append(cid)
                batch_docs.append(chunk)
                batch_metas.append({
                    "entity_id": entity_id,
                    "title": title,
                    "chunk_index": idx,
                    "n_chunks": n_chunks,
                    "source": os.path.basename(jsonl_path),
                })

            running_chunk_total += n_chunks
            p_docs.update(1)
            p_chunks.total = running_chunk_total
            p_chunks.refresh()

            if len(batch_ids) >= BATCH_SIZE:
                upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, BATCH_SIZE)
                # ADDED: Log batch progress
                logger.info(f"Upserted a batch of {len(batch_ids)} chunks. Total indexed: {p_chunks.n + len(batch_ids)}")
                p_chunks.update(len(batch_ids))
                batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

        if batch_ids:
            upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, BATCH_SIZE)
            # ADDED: Log final batch progress
            logger.info(f"Upserted final batch of {len(batch_ids)} chunks. Total indexed: {p_chunks.n + len(batch_ids)}")
            p_chunks.update(len(batch_ids))
            batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    finally:
        signal.signal(signal.SIGINT, old_handler)
        p_docs.close()
        p_chunks.close()

    if interrupted["flag"]:
        # CHANGED: from print to logger.info
        logger.info("Indexing interrupted after partial flush.")
    else:
        # CHANGED: from print to logger.info
        logger.info(f"Done. Successfully indexed {p_chunks.n} chunks from {total_docs} documents.")

if __name__ == "__main__":
    # ADDED: Configure logging to write to a file and the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout) # To still see output in the terminal
        ]
    )

    if not os.path.isfile(DOCUMENT_PATH):
        # CHANGED: from print to logger.error
        logger.error(f"File not found: {DOCUMENT_PATH}")
        sys.exit(1)
    index_jsonl(DOCUMENT_PATH)