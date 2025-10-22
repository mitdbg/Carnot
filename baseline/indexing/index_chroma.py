#!/usr/bin/env python3
# index_jsonl_to_chroma_fixed.py

import os, json, re, unicodedata, hashlib, signal, sys
from typing import List, Dict, Iterable
from tqdm import tqdm
from unidecode import unidecode

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = "./chroma_quest"
COLLECTION_NAME = "quest_documents"
DOCUMENT_PATH = "/orcd/home/002/joycequ/quest_data/documents.jsonl"

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 80
BATCH_SIZE = 256
CLEAR_COLLECTION = True
DEVICE = None

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
                print(f"[warn] Skipping malformed JSON on line {idx}: {e}", file=sys.stderr)

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
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    embed_fn = STEmbeddingFn(model_name=MODEL_NAME, device=DEVICE)

    if CLEAR_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[info] Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    tokenizer = build_tokenizer(MODEL_NAME)

    total_docs = sum(1 for _ in read_jsonl(jsonl_path))
    docs_iter = read_jsonl(jsonl_path)

    p_docs = tqdm(total=total_docs, desc="Documents processed", unit="doc")
    p_chunks = tqdm(total=0, desc="Chunks indexed", unit="chunk")

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict] = []
    running_chunk_total = 0

    interrupted = {"flag": False}
    def handle_sigint(sig, frame):
        interrupted["flag"] = True
        print("\n[warn] Interrupt received. Flushing pending batch before exit...", file=sys.stderr)

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
                p_chunks.update(len(batch_ids))
                batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

        if batch_ids:
            upsert_in_batches(collection, batch_ids, batch_docs, batch_metas, BATCH_SIZE)
            p_chunks.update(len(batch_ids))
            batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    finally:
        signal.signal(signal.SIGINT, old_handler)
        p_docs.close()
        p_chunks.close()

    if interrupted["flag"]:
        print("[info] Indexing interrupted after partial flush.", file=sys.stderr)
    else:
        print("Done.")

if __name__ == "__main__":
    if not os.path.isfile(DOCUMENT_PATH):
        print(f"[error] File not found: {DOCUMENT_PATH}", file=sys.stderr)
        sys.exit(1)
    index_jsonl(DOCUMENT_PATH)
