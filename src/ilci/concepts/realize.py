# src/ILCI/concepts/realize.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Literal, Union, TypedDict, Iterable, Set
from dataclasses import dataclass
import re
import chromadb

# ---------- Types ----------
class Concept(TypedDict):
    id: str
    name: str
    aliases: List[str]
    definition: Optional[str]

# realizer(chunk_text, concepts_subset) -> {normalized_key: value|None}
Realizer = Callable[[str, List[Concept]], Dict[str, Union[str, None]]]

# ---------- Config ----------
@dataclass
class RealizeConfig:
    persist_dir: str
    collection: str
    page_size: int = 1000
    sentinel_unknown: str = "unknown"
    key_conflict_policy: Literal["prefer_existing", "prefer_new", "merge"] = "prefer_new"

# ---------- Helpers ----------
_key_re = re.compile(r"[^a-z0-9_]+")

def keyify(name: str) -> str:
    """Normalize a human label into a stable metadata key (snake_case)."""
    s = name.strip().lower().replace(" ", "_")
    s = _key_re.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_key"

def _get_collection(cfg: RealizeConfig):
    client = chromadb.PersistentClient(path=cfg.persist_dir)
    return client.get_collection(name=cfg.collection)

def _paginate_get(col, page_size: int) -> Iterable[Dict[str, List]]:
    offset = 0
    while True:
        page = col.get(include=["ids", "documents", "metadatas"], limit=page_size, offset=offset)
        ids = page.get("ids") or []
        if not ids:
            break
        yield page
        offset += len(ids)

def _resolve_key_conflicts(
    existing: Dict[str, object],
    patch: Dict[str, object],
    policy: Literal["prefer_existing", "prefer_new", "merge"],
) -> Dict[str, object]:
    if policy == "prefer_existing":
        out = dict(patch); out.update(existing); return out
    if policy == "prefer_new":
        out = dict(existing); out.update(patch); return out
    # merge
    out = dict(existing)
    for k, v in patch.items():
        if k not in out:
            out[k] = v
            continue
        if out[k] == v:
            continue
        prev = out[k]
        if not isinstance(prev, list):
            prev = [prev]
        if isinstance(v, list):
            for item in v:
                if item not in prev:
                    prev.append(item)
        else:
            if v not in prev:
                prev.append(v)
        out[k] = prev
    return out

# ---------- Public API ----------
def realize_concepts_on_chunks(
    persist_dir: str,
    collection: str,
    concepts: List[Concept],
    realizer: Realizer,  # LLM/rules that can fill multiple keys in ONE call
    key_conflict_policy: Literal["prefer_existing", "prefer_new", "merge"] = "prefer_new",
    page_size: int = 1000,
    sentinel_unknown: str = "unknown",
) -> int:
    """
    Only realize *newly added* keys:
      - Compute normalized keys from `concepts`.
      - For each chunk, find which of those keys are *missing* from metadata.
      - If none missing, skip the chunk; otherwise call `realizer` ONCE with only those missing concepts.
      - Convert None -> sentinel; apply conflict policy; update metadata only.
    Returns number of chunks updated.
    """
    cfg = RealizeConfig(
        persist_dir=persist_dir,
        collection=collection,
        page_size=page_size,
        sentinel_unknown=sentinel_unknown,
        key_conflict_policy=key_conflict_policy,
    )
    col = _get_collection(cfg)

    # Map normalized key -> Concept and a fast set for membership tests
    key_to_concept: Dict[str, Concept] = {}
    new_schema_keys: Set[str] = set()
    for c in concepts:
        k = keyify(c["name"])
        # If duplicates map to first occurrence; you can change policy if desired
        if k not in key_to_concept:
            key_to_concept[k] = c
            new_schema_keys.add(k)

    total_updated = 0

    for page in _paginate_get(col, cfg.page_size):
        ids = page["ids"]
        docs = page["documents"]
        metas = page["metadatas"]

        out_metas: List[Dict[str, object]] = []
        out_ids: List[str] = []

        for cid, text, meta in zip(ids, docs, metas):
            meta = dict(meta or {})
            existing_keys = set(meta.keys())

            # Compute keys we need to realize *for this chunk* (missing only)
            missing_keys = [k for k in new_schema_keys if k not in existing_keys]
            if not missing_keys:
                # Nothing new to fill for this chunk
                continue

            # Build the subset of Concept objects for these missing keys
            subset_concepts = [key_to_concept[k] for k in missing_keys]

            # ONE call to the LLM/rules to fill all missing keys for this chunk
            patch = realizer(text, subset_concepts) or {}

            # Normalize: include only the missing keys we asked for; None -> sentinel
            filtered_patch: Dict[str, object] = {}
            for k in missing_keys:
                v = patch.get(k) or patch.get(keyify(key_to_concept[k]["name"]))  # accept either normalized or raw
                filtered_patch[k] = (v if v is not None else cfg.sentinel_unknown)

            # Merge into metadata with chosen conflict policy (rare here since keys are "missing")
            new_meta = _resolve_key_conflicts(meta, filtered_patch, policy=cfg.key_conflict_policy)

            out_ids.append(cid)
            out_metas.append(new_meta)

        if out_ids:
            # Update metadata only; embeddings/index unchanged
            col.update(ids=out_ids, metadatas=out_metas)
            total_updated += len(out_ids)

    return total_updated
