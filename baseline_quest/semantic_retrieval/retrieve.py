# SQLite compatibility
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("pysqlite3 not found, using system sqlite3. This might fail on some systems.")


import os
import json
from typing import Iterable, Dict
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_FIRST_512 = True
INCLUDE_CHUNKS = False # True --> output {title, chunk}. False --> output just title.

if INDEX_FIRST_512:
    PERSIST_DIR = "./chroma_quest_limited"
    COLLECTION_NAME = "quest_documents_limited"
    INPUT_QUERIES_PATH = "data/train_subset3.jsonl"
    OUTPUT_PREDICTIONS_PATH = "pred_unranked_limited.jsonl"
else:
    PERSIST_DIR = "./chroma_quest"
    COLLECTION_NAME = "quest_documents"
    INPUT_QUERIES_PATH = "train_subset.jsonl"
    OUTPUT_PREDICTIONS_PATH = "pred_unranked.jsonl"

DEVICE = None  # or "cuda"

def read_jsonl(path: str) -> Iterable[Dict]:
    """Reads a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def retrieve(queries_path: str, output_path: str):
    print(f"\n=== Retrieval Mode ===")
    print(f"INDEX_FIRST_512 = {INDEX_FIRST_512}")
    print(f"INCLUDE_CHUNKS = {INCLUDE_CHUNKS}")

    if not os.path.exists(PERSIST_DIR):
        print(f"Error: ChromaDB directory '{PERSIST_DIR}' does not exist.")
        return

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Loaded collection '{COLLECTION_NAME}' with {collection.count()} chunks.")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    queries = list(read_jsonl(queries_path))
    print(f"Loaded {len(queries)} queries.")

    # retrieve more chunks for full mode
    n_to_retrieve = 100 if INDEX_FIRST_512 else 200

    include_fields = ["metadatas"]
    if INCLUDE_CHUNKS:
        include_fields.append("documents")

    with open(output_path, "w", encoding="utf-8") as f_out:

        for item in tqdm(queries, desc="Retrieving…"):
            query_text = item["query"]

            # embed
            query_embedding = model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            # query chroma
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_to_retrieve,
                include=include_fields
            )

            metas = results["metadatas"][0]

            # if chunks are included, read them
            docs_text = results["documents"][0] if INCLUDE_CHUNKS else None

            # first 512 tokens, return title
            if INDEX_FIRST_512 and not INCLUDE_CHUNKS:
                top_docs = [meta["title"] for meta in metas][:100]

            # first 512 tokens, return (title, chunk)
            elif INDEX_FIRST_512 and INCLUDE_CHUNKS:
                top_docs = []
                for meta, chunk in zip(metas, docs_text):
                    top_docs.append({
                        "title": meta.get("title", "No Title"),
                        "chunk": chunk
                    })

            # full document, deduplicated titles
            elif not INDEX_FIRST_512 and not INCLUDE_CHUNKS:
                seen = set()
                ranked = []
                for meta in metas:
                    t = meta["title"]
                    if t not in seen:
                        seen.add(t)
                        ranked.append(t)
                top_docs = ranked[:100]

            # full document, non-deduplicated titles/chunks
            else:
                top_docs = []
                for meta, chunk in zip(metas, docs_text):
                    top_docs.append({
                        "title": meta.get("title", "No Title"),
                        "chunk": chunk
                    })

            # write result
            prediction = {"query": query_text, "docs": top_docs}
            f_out.write(json.dumps(prediction) + "\n")

    print(f"\nDone. Wrote: {output_path}")

if __name__ == "__main__":
    if not os.path.isfile(INPUT_QUERIES_PATH):
        print(f"Error: Missing query file: {INPUT_QUERIES_PATH}")
    else:
        retrieve(INPUT_QUERIES_PATH, OUTPUT_PREDICTIONS_PATH)
