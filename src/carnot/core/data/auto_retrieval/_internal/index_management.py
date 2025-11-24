# All the “instance-optimized auto-retrieval” / ILCI / metadata stuff lives here.
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
import os
import logging
import chromadb
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Optional
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import dspy

from ..config import Config
from ..types import Query

logger = logging.getLogger(__name__)

# ---------- Abstract interfaces (ABCs) ----------

class BaseVectorIndex(ABC):
    """Abstract interface for dense vector similarity search."""

    @abstractmethod
    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Ingest documents into the vector index."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        include: Sequence[str],
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Sequence[str], Sequence[str], Sequence[Mapping[str, Any]], Sequence[float]]:
        """Return (ids, documents, metadatas, distances) for the top-k nearest neighbors."""
        pass


class BaseKeywordIndex(ABC):
    """Abstract interface for keyword search."""

    @abstractmethod
    def filter(self, terms: Sequence[str], top_k: int | None = None) -> Sequence[str]:
        """Return doc_ids that satisfy the given keywords."""
        pass



class BaseMetadataStore(ABC):
    """Abstract interface for structural and semantic metadata tables."""

    @abstractmethod
    def register_schema(self, keys: Iterable[str]) -> None:
        """Register the global list of known metadata keys (schema)."""
        pass

    @abstractmethod
    def get_schema(self) -> Sequence[str]:
        """Return the current list of registered schema keys."""
        pass

    @abstractmethod
    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """Insert or update metadata for a document."""
        pass

    @abstractmethod
    def filter(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """Return doc_ids that satisfy structured metadata predicates."""
        pass


class BaseDocumentCatalog(ABC):
    """Abstract interface for the source-of-truth document store."""

    @abstractmethod
    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Upsert documents (text + base metadata) into the store."""
        pass

    @abstractmethod
    def get_documents(self, doc_ids: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        """Retrieve documents by ID (must return text + metadata)."""
        pass

    @abstractmethod
    def list_document_ids(self) -> Sequence[str]:
        """List all known document IDs."""
        pass


class BaseConceptGenerator(ABC):
    """Abstract component that learns semantic concepts from corpus + query log."""

    @abstractmethod
    def fit(
        self,
        docs: Iterable[Mapping[str, Any]],
        query_log: Iterable[Query],
    ) -> None:
        """Learn candidate concepts from documents and query traces."""
        pass

    @abstractmethod
    def concept_map(
        self,
        docs: Iterable[Mapping[str, Any]],
        concept_vocabulary: Optional[List[str]] = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return: mapping doc_id -> { 'concept:Name': 'Value', ... }"""
        pass


class BaseIndexManager(ABC):
    """Abstract manager for building and exposing an index portfolio."""

    @abstractmethod
    def bootstrap(self) -> None:
        """Run offline analysis to design and build indexes."""
        pass

    @abstractmethod
    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Ingest documents into all relevant indexes and metadata stores."""
        pass

    @abstractmethod
    def get_vector_index(self) -> BaseVectorIndex:
        """Return the underlying vector index handle."""
        pass

    @abstractmethod
    def get_keyword_index(self) -> BaseKeywordIndex:
        """Return the underlying keyword index handle."""
        pass

    @abstractmethod
    def get_metadata_store(self) -> BaseMetadataStore:
        """Return the underlying metadata store."""
        pass


# ---------- Concrete Chroma-backed implementations ----------

# ---- Embedding function for Chroma ----

class STEmbeddingFn:
    """SentenceTransformer-based embedding function for Chroma."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a batch of texts for Chroma."""
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


# ---- Chroma-backed catalog + vector index (one class, one collection) ----

class ChromaVectorIndex(BaseVectorIndex, BaseDocumentCatalog):
    """
    Dense vector index + document catalog backed by a single Chroma collection.

    Each row in the collection corresponds to a *chunk* (or whole document),
    with:
      - id
      - embedding   (managed by Chroma via embedding_function)
      - document    (chunk text)
      - metadata    (JSON dict with dataset + pipeline fields)
    """

    def __init__(self, config: Config) -> None:
        self._persist_dir = config.chroma_persist_dir
        self._collection_name = config.chroma_collection_name
        self._embedding_model_name = getattr(
            config, "embedding_model_name", "BAAI/bge-small-en-v1.5"
        )
        # Match legacy script behavior: larger batches for first-512 mode
        self._batch_size = 2048 if getattr(config, "index_first_512", False) else 256

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = STEmbeddingFn(self._embedding_model_name)

        # Chroma will call _embed_fn when we upsert documents.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(f"Initialized ChromaVectorIndex (collection={self._collection_name}, persist_dir={self._persist_dir})")
        logger.info(f"ChromaVectorIndex: using upsert batch_size={self._batch_size} (index_first_512={getattr(config, 'index_first_512', False)})")

    # ---------- BaseDocumentCatalog API ----------

    def reset_collection(self) -> None:
        """
        Delete and recreate the underlying Chroma collection.
        """
        logger.info(f"ChromaVectorIndex: resetting collection '{self._collection_name}'.")
        try:
            # Delete existing collection if present
            self._client.delete_collection(self._collection_name)
        except Exception:
            # Ignore if collection does not exist
            pass
        # Recreate collection with the same embedding function
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
        )

    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """
        Add chunks/documents to Chroma.

        Each element in `docs` is expected to be:
          {
            "id": <chunk_id or doc_id>,
            "text": <chunk text>,
            "metadata": { ... arbitrary JSON-serializable fields ... }
          }
        """
        ids_batch: List[str] = []
        texts_batch: List[str] = []
        metas_batch: List[Mapping[str, Any]] = []
        pos_by_id: Dict[str, int] = {}
        total = 0

        def flush() -> None:
            nonlocal ids_batch, texts_batch, metas_batch, pos_by_id, total
            if not ids_batch:
                return
            self._collection.upsert(ids=ids_batch, documents=texts_batch, metadatas=metas_batch)
            total += len(ids_batch)
            logger.info(f"ChromaVectorIndex: upserted {len(ids_batch)} documents (total={total}).")
            ids_batch = []
            texts_batch = []
            metas_batch = []
            pos_by_id = {}

        for d in docs:
            doc_id = str(d.get("id") or d.get("doc_id"))
            text = d.get("text", "")
            metadata = dict(d.get("metadata", {}))

            if not doc_id or not text:
                continue

            if doc_id in pos_by_id:
                pos = pos_by_id[doc_id]
                texts_batch[pos] = text
                metas_batch[pos] = metadata
            else:
                pos_by_id[doc_id] = len(ids_batch)
                ids_batch.append(doc_id)
                texts_batch.append(text)
                metas_batch.append(metadata)

            if len(ids_batch) >= self._batch_size:
                flush()

        flush()

    def get_documents(self, doc_ids: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        """Retrieve documents (text + metadata) by ID from Chroma."""
        if not doc_ids:
            return []

        results = self._collection.get(
            ids=list(doc_ids),
            include=["documents", "metadatas"],
        )

        found_ids = results.get("ids", [])
        found_texts = results.get("documents", [])
        found_metas = results.get("metadatas", [])

        out: List[Mapping[str, Any]] = []
        for i, doc_id in enumerate(found_ids):
            out.append(
                {
                    "id": doc_id,
                    "text": found_texts[i],
                    "metadata": found_metas[i] if found_metas else {},
                }
            )
        return out

    def list_document_ids(self) -> Sequence[str]:
        """List all stored chunk IDs."""
        results = self._collection.get(include=[])
        return results.get("ids", [])

    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Update metadata for an existing document in Chroma.
        Performs a read-modify-write to preserve existing fields.
        """
        # Fetch existing metadata
        results = self._collection.get(ids=[doc_id], include=["metadatas"])
        existing_metas = results.get("metadatas")
        
        current_meta = {}
        if existing_metas and len(existing_metas) > 0 and existing_metas[0]:
            current_meta = existing_metas[0]

        # Merge new metadata
        current_meta.update(metadata)

        # Write back
        self._collection.update(
            ids=[doc_id],
            metadatas=[current_meta]
        )

    def filter_by_metadata(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """
        Return doc_ids that satisfy the given metadata predicates (AND).
        """
        if not predicates:
            return self.list_document_ids()

        # Chroma expects 'where' clause
        # If single predicate: {"key": value}
        # If multiple: {"$and": [{"key": val}, ...]}
        
        where_clause: Dict[str, Any] = {}
        items = list(predicates.items())
        
        if len(items) == 1:
            where_clause = {items[0][0]: items[0][1]}
        else:
            where_clause = {"$and": [{k: v} for k, v in items]}

        results = self._collection.get(where=where_clause, include=[])
        return results.get("ids", [])
    
    def filter_by_keyword(
        self,
        terms: Sequence[str],
        top_k: int | None = None,
    ) -> Sequence[str]:
        """
        Return doc_ids whose *document text* matches the given keywords.
        This uses Chroma's `where_document` + `$contains` for full-text search.

        - Currently uses AND semantics: all terms must appear in the text.
        - If `top_k` is None, returns all matching ids (subject to Chroma limits).
        """
        # Normalize terms
        clean_terms = [t.strip() for t in terms if t and t.strip()]
        if not clean_terms:
            return []

        # Build where_document:
        # 1 term: {"$contains": "foo"}
        # >1 term: {"$and": [{"$contains": "foo"}, {"$contains": "bar"}]}
        if len(clean_terms) == 1:
            where_document: dict[str, Any] = {"$contains": clean_terms[0]}
        else:
            where_document = {"$and": [{"$contains": t} for t in clean_terms]}

        # Chroma's get() supports where_document + limit
        results = self._collection.get(
            where_document=where_document,
            limit=top_k,           # None means "no explicit limit"
            include=[],            # we only need ids
        )
        return results.get("ids", []) or []

    # ---------- BaseVectorIndex API ----------

    def query(self, query_embedding: Sequence[float], top_k: int, include: Sequence[str] = ["documents", "metadatas", "distances"], filters: Optional[Mapping[str, Any]] = None) -> Tuple[Sequence[str], Sequence[str], Sequence[Mapping[str, Any]], Sequence[float]]:
        """
        Query Chroma for nearest neighbors given a precomputed embedding.

        Returns a list of (id, distance) pairs.
        """
        logger.info(f"ChromaVectorIndex: querying with top_k={top_k}, filters={filters}")
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=include,
            where=filters,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            return ([], [], [], [])

        return (ids, documents, metadatas, distances)


# ---- Simple keyword index stub (kept minimal for now) ----
class ChromaKeywordIndex(BaseKeywordIndex):
    """
    Thin façade over ChromaVectorIndex's keyword filter.
    """

    def __init__(self, vector_index: ChromaVectorIndex) -> None:
        self._vector_index = vector_index

    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        # Vector index already handles ingest; nothing to do here.
        return

    def filter(self, terms: Sequence[str], top_k: int | None = None) -> Sequence[str]:
        """Keyword filter façade over the vector index."""
        return self._vector_index.filter_by_keyword(terms, top_k=top_k)


# ---- Structured metadata table with uniform schema ----

class TableMetadataStore(BaseMetadataStore):
    """
    Metadata store backed directly by Chroma.
    
    Instead of maintaining an in-memory dict, this delegates storage and 
    filtering to the underlying Chroma collection.
    """

    def __init__(self, vector_index: ChromaVectorIndex) -> None:
        self._vector_index = vector_index

    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Insert or update metadata for a document via Chroma.
        """
        self._vector_index.upsert_metadata(doc_id, metadata)

    def filter(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """Filter documents using Chroma's 'where' clause."""
        return self._vector_index.filter_by_metadata(predicates)

    def list_metadata_keys(self, sample_size: int = 30) -> Sequence[str]:
        """
        Best-effort list of metadata keys present in the collection.

        This inspects up to `sample_size` documents from Chroma and unions
        their metadata keys. Under our design, all docs are written with
        the same set of fields, so this approximates the global schema.
        """
        # Get up to `sample_size` IDs
        all_ids = self._vector_index.list_document_ids()
        if not all_ids:
            return []

        sample_ids = all_ids[:sample_size]
        docs = self._vector_index.get_documents(sample_ids)

        keys: set[str] = set()
        for d in docs:
            meta = d.get("metadata", {}) or {}
            keys.update(meta.keys())

        return sorted(keys)


# ---------- Internal helpers for concept generation ----------

def _parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; attempt simple salvage if extra text appears."""
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Heuristic salvage: extract first [...] and parse
    if "[" in raw and "]" in raw:
        candidate = raw[raw.find("["): raw.rfind("]") + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Numbered list format: [1] «concept» or [1] "concept"
    numbered_pattern = r"\[\d+\]\s*[«\"]([^»\"]+)[»\"]"
    matches = re.findall(numbered_pattern, raw)
    if matches:
        return [m.strip() for m in matches]

    # Fallback: treat whole string as one concept
    return [raw]


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Case-insensitive deduplication while preserving original order."""
    seen = set()
    result: List[str] = []
    for item in items:
        norm = item.strip().lower()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        result.append(item.strip())
    return result

class PerQueryConceptSignature(dspy.Signature):
    """
    Generate mid-granularity concepts for a single query.
    """
    query = dspy.InputField(
        desc="Natural language query with implicit set operations."
    )
    concepts = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of strings (no prose before/after). "
            "Each string is ONE self-contained, Boolean-friendly concept."
        )
    )


class PerQueryConceptModel(dspy.Module):
    """
    LLM wrapper that maps a single query → per-query concept list.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(PerQueryConceptSignature)

        # Optional: built-in few-shot examples (generic, not dataset-specific).
        self._few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                concepts='['
                         '"Vertebrates of Kolombangara",'
                         '"Birds on the New Georgia Islands group",'
                         '"Vertebrates of the Western Province (Solomon Islands)",'
                         '"Birds of the Solomon Islands"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='['
                         '"Trees of Africa",'
                         '"Flora of South Africa",'
                         '"Flora of the South-Central Pacific",'
                         '"Trees in the Pacific",'
                         '"Coastal trees"'
                         ']'
            ).with_inputs("query"),
            dspy.Example(
                query="2010s adventure films set in the Southwestern United States but not in California",
                concepts='['
                         '"Adventure films",'
                         '"2010s films",'
                         '"Films set in the U.S. Southwest",'
                         '"Films set in California"'
                         ']'
            ).with_inputs("query"),
        ]

    def forward(self, query: str) -> List[str]:
        """Run the LLM and return a parsed list of concepts."""
        result = self._predict(query=query, demos=self._few_shot_examples)
        raw = getattr(result, "concepts", "") or ""
        return _parse_concept_list(raw)


class BatchFinalConceptSignature(dspy.Signature):
    """
    Directly generate final abstract concepts from a list of queries.
    """
    queries = dspy.InputField(
        desc="A list of natural language queries with implicit set operations."
    )
    final_concepts = dspy.OutputField(
        desc="Return ONLY a JSON list of UNIQUE short noun phrases."
    )


class BatchFinalConceptModel(dspy.Module):
    """
    ONE-SHOT: list of queries → deduped list of final concepts.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(BatchFinalConceptSignature)
        self._few_shot_examples = [
            dspy.Example(
                queries=[
                    "Birds of Kolombangara or of the Western Province (Solomon Islands)",
                    "Trees of South Africa that are also in the south-central Pacific",
                    "2010s adventure films set in the Southwestern United States but not in California",
                ],
                final_concepts=[
                    "bird geographic distribution",
                    "plant geographic distribution",
                    "film genre",
                    "film setting and location",
                ],
            ).with_inputs("queries"),
        ]

    def forward(self, queries: List[str]) -> List[str]:
        """Run the LLM and return a parsed, deduped list of final concepts."""
        result = self._predict(queries=queries, demos=self._few_shot_examples)
        raw = getattr(result, "final_concepts", "") or ""
        parsed = _parse_concept_list(raw)
        return _dedupe_preserve_order(parsed)


class ClusterCentroidSignature(dspy.Signature):
    """
    Generate a compact centroid (short noun phrase) for a cluster of related concepts.
    """
    concepts = dspy.InputField(
        desc="A list of short concept strings that belong to ONE semantic cluster."
    )
    centroid = dspy.OutputField(
        desc=(
            "Return ONLY a SINGLE, short, singular noun phrase (2-5 words) "
            "describing the cluster."
        )
    )


class ClusterCentroidModel(dspy.Module):
    """
    LLM wrapper: cluster of concepts → centroid label.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(ClusterCentroidSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=[
                    "Birds of the Pacific Islands",
                    "Birds of North America",
                    "Birds found in Central Africa",
                ],
                centroid="avian geographic region",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Horror films",
                    "Historical films",
                    "Films set in the future",
                    "Black-and-white films",
                ],
                centroid="film genre or style",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "1990s films",
                    "1988 films",
                    "Films released in 1975",
                    "Early 1960s films",
                ],
                centroid="film release period",
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        """Run the LLM and return a centroid label."""
        # Format as a bullet list for nicer prompting
        concepts_str = "\n".join(f"- {c}" for c in concepts)
        result = self._predict(concepts=concepts_str, demos=self._few_shot_examples)
        centroid = (getattr(result, "centroid", "") or "").strip()
        return centroid

class ConceptGenerationMode(Enum):
    """Supported concept generation strategies."""

    TWO_STAGE = "two_stage"   # per-query → cluster → centroid (default)
    DIRECT = "direct"         # direct concepts from list of queries
    

# ---------- LLM-based concept generation ----------

@dataclass
class LLMConceptGenerator(BaseConceptGenerator):
    """
    LLM-based component that learns and assigns workload-specific concepts.

    This wraps two strategies:

    - TWO_STAGE (default): per-query intermediate concepts → clustering →
      centroid labels (final concepts).
    - DIRECT: a single LLM pass over the full query list to get final concepts.
    """

    config: Config
    mode: ConceptGenerationMode = ConceptGenerationMode.TWO_STAGE
    n_clusters: int = 50
    embedding_model_name: str = "all-MiniLM-L6-v2"

    def __init__(self, config: Config) -> None:
        """Initialize the concept generator from configuration."""
        # Extract mode and hyperparameters from config if present
        mode_str = getattr(config, "concept_generation_mode", ConceptGenerationMode.TWO_STAGE.value)
        try:
            mode = ConceptGenerationMode(mode_str)
        except ValueError:
            mode = ConceptGenerationMode.TWO_STAGE

        n_clusters = getattr(config, "concept_cluster_count", 50)
        embedding_model_name = getattr(config, "concept_embedding_model", "all-MiniLM-L6-v2")

        # dataclass-style manual initialization
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "n_clusters", n_clusters)
        object.__setattr__(self, "embedding_model_name", embedding_model_name)

        # LLM-based submodules
        object.__setattr__(self, "_per_query_model", PerQueryConceptModel())
        object.__setattr__(self, "_batch_final_model", BatchFinalConceptModel())
        object.__setattr__(self, "_centroid_model", ClusterCentroidModel())

        # Learned vocabulary of final concepts
        object.__setattr__(self, "_concept_vocabulary", [])  # type: ignore[var-annotated]

    def fit(
        self,
        docs: Iterable[Mapping[str, Any]],
        query_log: List[str],
    ) -> None:
        """
        Infer a workload-specific vocabulary of semantic concepts.

        NOTE: docs are currently ignored; concepts are inferred only from
        the query log.
        """
        logger.info(f"LLMConceptGenerator: fitting on {len(query_log)} queries (mode={self.mode}).")

        if not query_log:
            object.__setattr__(self, "_concept_vocabulary", [])
            return

        if self.mode is ConceptGenerationMode.TWO_STAGE:
            concepts = self._fit_two_stage(query_log)
        else:
            concepts = self._fit_direct(query_log)

        logger.info(f"LLMConceptGenerator: learned {len(concepts)} concepts.")
        object.__setattr__(self, "_concept_vocabulary", concepts)

    def concept_map(
        self,
        docs: Iterable[Mapping[str, Any]],
        concept_vocabulary: Optional[List[str]] = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        """
        Given docs and a (learned) concept vocabulary, extract per-doc values
        for each concept using Palimpzest and return a mapping:

            doc_id -> { "concept:<concept_name>": List[str] }

        This mapping is designed to be written directly into the metadata
        store (and then into ChromaDB), where each "concept:*" field is
        a multi-valued metadata column.
        The where clause for a semantic filter would be: where = {"concept:film genre": {"$contains": "sci-fi"}}
        """
        # 1) Resolve concept vocabulary
        if concept_vocabulary is None:
            concept_vocabulary = self._concept_vocabulary
        if not concept_vocabulary:
            logger.info("LLMConceptGenerator.concept_map: empty concept vocabulary; nothing to do.")
            return {}

        # 2) Materialize docs and extract ids + text
        docs_list: List[Mapping[str, Any]] = list(docs)
        if not docs_list:
            logger.info("LLMConceptGenerator.concept_map: no docs provided.")
            return {}

        pz_rows: List[Dict[str, Any]] = []
        for d in docs_list:
            doc_id = str(d.get("id"))
            if not doc_id:
                continue

            # Adjust this field name to how your chunks store text
            text = d.get("text") or d.get("content") or ""
            if not text:
                # Skip empty-text docs; nothing to map
                continue

            pz_rows.append({"id": doc_id, "text": text})

        if not pz_rows:
            logger.info("LLMConceptGenerator.concept_map: no docs with text.")
            return {}

        # 3) Build PZ column specs for all concepts
        cols_spec: List[Dict[str, Any]] = []
        for concept in concept_vocabulary:
            col_name = f"concept:{concept}"
            col_spec = self._build_pz_col_spec(concept_name=concept, col_name=col_name)
            cols_spec.append(col_spec)

        # 4) Create PZ dataset
        dataset = pz.MemoryDataset(id="concept-assignment", vals=pz_rows)

        # 5) Run semantic map to extract concept values
        dataset = dataset.sem_map(cols=cols_spec)
        output = dataset.run(max_quality=True)
        df = output.to_df()

        # 6) Post-process PZ outputs into multi-valued metadata
        assignments: Dict[str, Dict[str, Any]] = {}

        concept_col_names = [f"concept:{c}" for c in concept_vocabulary]

        for _, row in df.iterrows():
            doc_id = str(row.get("id"))
            if not doc_id:
                continue

            concept_attrs: Dict[str, Any] = {}

            for col_name in concept_col_names:
                if col_name not in row:
                    continue

                raw_val = row[col_name]
                if raw_val is None:
                    values: List[str] = []
                else:
                    # Normalize to string
                    s = str(raw_val).strip()
                    if not s:
                        values = []
                    else:
                        # Heuristic split: comma / semicolon separated labels
                        parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
                        values = parts or [s]

                # Store as list-of-strings for multi-valued concept fields
                concept_attrs[col_name] = values

            if concept_attrs:
                assignments[doc_id] = concept_attrs

        return assignments

    def _build_pz_col_spec(self, concept_name: str, col_name: str) -> Dict[str, Any]:
        """
        Build a PZ column specification for a given concept.

        For now, we generate a simple deterministic description; you can
        later replace this with an actual LLM call that tailors the `desc`.
        """
        # TODO (optional): call an LLM here to determine "type" of the column and generate richer descriptions
        desc = (
            f"The {concept_name} associated with this text chunk. "
            f"Return one or more short labels that best describe the {concept_name}."
            f"If the document does not relate to the concept, set the field to None."
        )

        return {
            "name": col_name,
            "type": str,  # PZ will output strings; we'll post-process into lists.
            "desc": desc,
        }


    def generate_from_queries(self, queries: List[str]) -> List[str]:
        """
        Convenience method: given a list of raw query strings, learn and
        return the concept vocabulary.

        This bypasses the Query dataclass and does not touch docs.
        """
        self.fit(docs=[], query_log=queries)
        return list(self._concept_vocabulary)

    def get_concept_vocabulary(self) -> List[str]:
        """Return the learned concept vocabulary (final concepts)."""
        return list(self._concept_vocabulary)

    def _fit_two_stage(self, queries: List[str]) -> List[str]:
        """
        TWO-STAGE STRATEGY:
        1. LLM generates per-query intermediate concepts.
        2. Concepts are embedded and clustered.
        3. LLM generates a centroid (final concept) for each cluster.
        """
        # 1) Per-query concepts
        all_concepts: List[str] = []
        for query in queries:
            per_query_concepts = self._per_query_model(query)
            all_concepts.extend(per_query_concepts)

        all_concepts = _dedupe_preserve_order(all_concepts)
        logger.info(f"LLMConceptGenerator: generated {len(all_concepts)} intermediate concepts.")
        if not all_concepts:
            return []

        # 2) Embed + cluster concepts
        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(all_concepts, show_progress_bar=False)

        # If fewer concepts than clusters, reduce cluster count
        n_clusters = min(self.n_clusters, len(all_concepts))
        logger.info(f"LLMConceptGenerator: clustering into {n_clusters} clusters.")
        if n_clusters <= 0:
            return []

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters: Mapping[int, List[str]] = {}
        for concept, label in zip(all_concepts, labels):
            clusters.setdefault(int(label), []).append(concept)  # type: ignore[attr-defined]

        # 3) LLM centroid per cluster
        final_concepts: List[str] = []
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]
            if not members:
                continue
            centroid = self._centroid_model(members)
            if centroid:
                final_concepts.append(centroid)

        return _dedupe_preserve_order(final_concepts)

    def _fit_direct(self, queries: List[str]) -> List[str]:
        """
        DIRECT STRATEGY:
        Feed all queries to the LLM and ask for final concepts directly.
        """
        # You can choose to chunk queries if there are many; for now, send all.
        final_concepts = self._batch_final_model(queries)
        return _dedupe_preserve_order(final_concepts)


# ---- High-level index management pipeline ----

@dataclass
class IndexManagementPipeline(BaseIndexManager):
    """
    Orchestrates ingestion, metadata, concept generation, and indexes on Chroma.

    Flow:
      - add_documents: store chunks in Chroma + register base metadata in TableMetadataStore
      - enrich_documents: assign concepts, update metadata
    """

    document_catalog: BaseDocumentCatalog
    vector_index: BaseVectorIndex
    keyword_index: BaseKeywordIndex
    metadata_store: TableMetadataStore
    concept_generator: BaseConceptGenerator

    @classmethod
    def from_config(cls, config: Config) -> "IndexManagementPipeline":
        # Single Chroma-backed object used as both catalog and vector index
        chroma_index = ChromaVectorIndex(config)
        keyword = ChromaKeywordIndex(config)
        
        # Metadata store now wraps the vector index (Chroma)
        metadata = TableMetadataStore(chroma_index)
        
        concept_gen = LLMConceptGenerator(config)

        return cls(
            document_catalog=chroma_index,
            vector_index=chroma_index,
            keyword_index=keyword,
            metadata_store=metadata,
            concept_generator=concept_gen,
        )

    def bootstrap(self, query_log: Iterable[Query] = None) -> None:
        """
        Offline concept vocabulary learning (no docs needed yet).

        We just fit the concept generator on the query log.
        """
        logger.info("IndexManagementPipeline: bootstrapping...")
        q_log = query_log or []
        q_strings = [q.text for q in q_log] if q_log else []
        self.concept_generator.fit(docs=[], query_log=q_strings)
        logger.info("IndexManagementPipeline: bootstrap complete.")

    def add_documents(self, docs: Iterable[Mapping[str, Any]], reset: bool = False) -> None:
        """
        Ingest documents/chunks into the system (Chroma + metadata + keyword index).

        Each input doc is treated as a *chunk*:
          {
            "id": <chunk_id>,
            "text": <chunk text>,
            "metadata": { ... base fields from dataset / pipeline ... }
          }

        After this call:
          - Chroma has (id, embedding, document text, base metadata).
        """
        docs_list = list(docs)
        if not docs_list:
            return

        logger.info(f"IndexManagementPipeline: adding {len(docs_list)} documents.")

        # Optionally clear the Chroma collection before ingesting
        if reset and hasattr(self.document_catalog, "reset_collection"):
            logger.info("IndexManagementPipeline: clear_chroma_collection=True; resetting underlying Chroma collection...")
            self.document_catalog.reset_collection()

        # Source of truth for text + embeddings + raw metadata: Chroma
        self.document_catalog.add_documents(docs_list)

        # Keyword index (if implemented)
        self.keyword_index.add_documents(docs_list)

    def enrich_documents(self, doc_ids: Sequence[str]) -> None:
        """
        Run concept assignment and update metadata for EXISTING documents.

        After this call:
          - concept:* fields are added to TableMetadataStore (same schema for all docs).
        """
        if not doc_ids:
            return

        logger.info(f"IndexManagementPipeline: enriching {len(doc_ids)} documents.")
        # Fetch the actual chunk texts from Chroma (if needed by the concept generator)
        docs = self.document_catalog.get_documents(doc_ids)
        if not docs:
            return

        # Run concept assignment (produces a dict: doc_id -> { "concept:*": value, ... })
        assignments = self.concept_generator.concept_map(docs)

        # Update structured metadata store with concept fields
        for doc_id, concept_attrs in assignments.items():
            self.metadata_store.upsert_metadata(doc_id, concept_attrs)

    def get_vector_index(self) -> BaseVectorIndex:
        return self.vector_index

    def get_keyword_index(self) -> BaseKeywordIndex:
        return self.keyword_index

    def get_metadata_store(self) -> BaseMetadataStore:
        return self.metadata_store

    def get_index_manager(self) -> BaseIndexManager:
        return self
