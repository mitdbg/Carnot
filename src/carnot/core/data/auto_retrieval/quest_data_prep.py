from __future__ import annotations

import os
import re
import json
import hashlib
import logging
import unicodedata
from typing import List, Dict, Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_MODEL = "BAAI/bge-small-en-v1.5"


# -----------------------------
# Helpers copied from QUEST script
# -----------------------------

def normalize_title_slug(s: str) -> str:
    if not s:
        return "untitled"
    t = unicodedata.normalize("NFC", s).strip()
    from unidecode import unidecode
    t = unidecode(t)
    # normalize whitespace and strip weird chars
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Za-z0-9 _\-.]", "", t).strip().replace(" ", "_")
    return t or "untitled"


def stable_entity_id(title: str, text: str) -> str:
    slug = normalize_title_slug(title)
    h = hashlib.sha1((title + "\n" + (text or "")).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON on line {idx}: {e}")


def build_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def chunk_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    toks = tokenizer.encode(text, add_special_tokens=False)
    if not toks:
        return []

    chunks: List[str] = []
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


# -----------------------------
# QUEST document preparation
# -----------------------------

def prepare_quest_documents(
    jsonl_path: str,
    tokenizer=None,
    index_first_512: bool = True,
    chunk_tokens: int = 512,
    overlap_tokens: int = 80,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
) -> Iterable[Dict[str, Any]]:
    """
    For each JSON line (raw record), we:
      - Use `title` and `text` (with fallbacks) as in the original script.
      - Build a stable `entity_id` via `stable_entity_id(title, text)`.
      - Either:
          * index_first_512=True: single chunk of first `chunk_tokens` tokens
          * index_first_512=False: full token-based chunking with overlap
      - Yield dicts with:
          {
            "id": <chunk_id>,         # entity_id or entity_id__0000, etc.
            "text": <chunk_text>,     # chunk body
            "metadata": {
                "entity_id": ...,
                "title": ...,
                "chunk_index": ...,
                "n_chunks": ...,
                "source": <basename of jsonl_path>,
            },
          }
    """
    if tokenizer is None:
        tokenizer = build_tokenizer(tokenizer_model)

    filename = os.path.basename(jsonl_path)

    for raw in read_jsonl(jsonl_path):
        title = (raw.get("title") or "").strip() or "untitled"
        text = (raw.get("text") or raw.get("description") or "").strip()

        entity_id = stable_entity_id(title, text)

        if index_first_512:
            toks = tokenizer.encode(text, add_special_tokens=False)
            truncated = toks[:chunk_tokens]
            chunk_text_str = tokenizer.decode(truncated, skip_special_tokens=True).strip()

            if not chunk_text_str:
                # same fallback as in the original script
                chunk_text_str = title

            metadata = {
                "entity_id": entity_id,
                "title": title,
                "chunk_index": 0,
                "n_chunks": 1,
                "source": filename,
            }

            yield {
                "id": entity_id,   
                "text": chunk_text_str,
                "metadata": metadata,
            }

        else:
            # --- VERBOSE MODE: full token-based chunking with overlap ---
            chunks = chunk_by_tokens(text, tokenizer, chunk_tokens, overlap_tokens)
            if not chunks:
                chunks = [title]

            n_chunks = len(chunks)

            for idx, ch in enumerate(chunks):
                cid = f"{entity_id}__{idx:04d}"

                metadata = {
                    "entity_id": entity_id,
                    "title": title,
                    "chunk_index": idx,
                    "n_chunks": n_chunks,
                    "source": filename,
                }

                yield {
                    "id": cid,
                    "text": ch,
                    "metadata": metadata,
                }
