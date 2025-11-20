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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple
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
    def query(self, query_embedding: Sequence[float], top_k: int, include: Sequence[str]) -> Sequence[Tuple[str, float]]:
        """Return (doc_id, score) tuples of the top-k nearest neighbors."""
        pass


class BaseKeywordIndex(ABC):
    """Abstract interface for keyword / inverted index search."""

    @abstractmethod
    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Ingest documents into the keyword index."""
        pass

    @abstractmethod
    def query(self, terms: Sequence[str], top_k: int) -> Sequence[Tuple[str, float]]:
        """Return (doc_id, score) tuples matching the given terms."""
        pass


class BaseConceptIndex(ABC):
    """Abstract interface for the inverted learned concept index (ILCI)."""

    @abstractmethod
    def materialize_concepts(self) -> None:
        """Materialize concept columns over the corpus."""
        pass

    @abstractmethod
    def select(self, concept_names: Sequence[str]) -> Sequence[str]:
        """Return doc_ids that satisfy the given concept predicates."""
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
    def assign_concepts(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Returns a mapping: doc_id -> {concept: concept_value}"""
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
    def get_concept_index(self) -> BaseConceptIndex:
        """Return the underlying concept index handle."""
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

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts for Chroma."""
        if not texts:
            return []
        embs = self.model.encode(
            texts,
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

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = STEmbeddingFn(self._embedding_model_name)

        # Chroma will call _embed_fn when we upsert documents.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(f"Initialized ChromaVectorIndex (collection={self._collection_name}, persist_dir={self._persist_dir})")

    # ---------- BaseDocumentCatalog API ----------

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
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Mapping[str, Any]] = []

        for d in docs:
            doc_id = str(d.get("id") or d.get("doc_id"))
            text = d.get("text", "")
            metadata = dict(d.get("metadata", {}))

            if not doc_id or not text:
                continue

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        if not ids:
            return

        logger.info(f"ChromaVectorIndex: upserting {len(ids)} documents.")
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

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

    # ---------- BaseVectorIndex API ----------

    def query(self, query_embedding: Sequence[float], top_k: int, include: Sequence[str] = ["documents", "metadatas", "distances"]) -> Tuple[Sequence[str], Sequence[str], Sequence[Mapping[str, Any]], Sequence[float]]:
        """
        Query Chroma for nearest neighbors given a precomputed embedding.

        Returns a list of (id, distance) pairs.
        """
        logger.info(f"ChromaVectorIndex: querying with top_k={top_k}")
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=include,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not ids:
            return []

        return (ids, documents, metadatas, distances)


# ---- Simple keyword index stub (kept minimal for now) ----

class ChromaKeywordIndex(BaseKeywordIndex):
    """Placeholder keyword / inverted index."""

    def __init__(self, config: Config) -> None:
        pass

    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
        """Index raw text and tokens for lexical retrieval (not implemented yet)."""
        return

    def query(self, terms: Sequence[str], top_k: int, include: Sequence[str]) -> Sequence[Tuple[str, float]]:
        """Return documents matching the given keywords (empty for now)."""
        return []


# ---- Structured metadata table with uniform schema ----

class TableMetadataStore(BaseMetadataStore):
    """
    Simple in-memory metadata store that enforces a uniform schema across all docs.

    - Every doc has the same set of keys (fields/attributes).
    - New keys are auto-registered, existing docs are backfilled with None.
    """

    def __init__(self, config: Config) -> None:
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._schema_keys: Set[str] = set()

    def register_schema(self, keys: Iterable[str]) -> None:
        """Register new keys into the global schema and backfill existing docs."""
        new_keys = set(keys) - self._schema_keys
        if not new_keys:
            return

        logger.info(f"TableMetadataStore: registering new schema keys: {sorted(list(new_keys))}")
        self._schema_keys.update(new_keys)

        # Backfill existing docs with None for the new keys
        for doc_id in self._metadata:
            for k in new_keys:
                if k not in self._metadata[doc_id]:
                    self._metadata[doc_id][k] = None

    def get_schema(self) -> Sequence[str]:
        """Return the current list of registered schema keys."""
        return sorted(self._schema_keys)

    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Insert or update metadata for a document.

        - Any new keys extend the global schema.
        - All documents are kept schema-consistent.
        - Concept fields (e.g., 'concept:genre') are treated the same as base fields.
        """
        incoming_keys = set(metadata.keys())
        new_keys = incoming_keys - self._schema_keys

        if new_keys:
            self.register_schema(new_keys)

        # Start with all current schema keys set to None
        record = {k: None for k in self._schema_keys}

        # Merge with existing record (if any)
        existing = self._metadata.get(doc_id, {})
        record.update(existing)

        # Merge in the new metadata
        record.update(metadata)

        self._metadata[doc_id] = record

    def filter(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """Filter documents using structured predicates (AND semantics)."""
        results: List[str] = []
        for doc_id, meta in self._metadata.items():
            ok = True
            for key, value in predicates.items():
                if meta.get(key) != value:
                    ok = False
                    break
            if ok:
                results.append(doc_id)
        return results

    def iter_all(self) -> Iterable[tuple[str, Dict[str, Any]]]:
        """Iterate over all (doc_id, metadata) pairs."""
        return self._metadata.items()


# ---- Inverted index over generated concepts ----

class InvertedConceptIndex(BaseConceptIndex):
    """
    Inverted index over learned semantic concepts.

    We treat concept fields as normal metadata keys with a 'concept:' prefix.
    Example: metadata["concept:film genre"] = "horror"
    Becomes an index entry under "film genre:horror".
    """

    def __init__(self, config: Config, metadata_store: TableMetadataStore) -> None:
        self._metadata_store = metadata_store
        self._posting_lists: Dict[str, List[str]] = defaultdict(list)

    def materialize_concepts(self) -> None:
        """Scan metadata and build posting lists for all concept:* fields."""
        logger.info("InvertedConceptIndex: starting materialization...")
        self._posting_lists.clear()

        for doc_id, meta in self._metadata_store.iter_all():
            for key, value in meta.items():
                if not key.startswith("concept:"):
                    continue
                if not value:
                    continue

                concept_name = key[len("concept:") :].strip()
                val_str = str(value).strip().lower()
                if not concept_name or not val_str:
                    continue

                index_key = f"{concept_name}:{val_str}"
                self._posting_lists[index_key].append(doc_id)

        logger.info(f"InvertedConceptIndex: materialized {len(self._posting_lists)} concepts.")

    def select(self, concept_predicates: Sequence[str]) -> Sequence[str]:
        """
        Return documents that satisfy all given concept predicates.

        Predicates are of the form "ConceptName:Value", e.g.:
          ["film genre:horror", "release period:1990s"]
        """
        if not concept_predicates:
            return []

        sets: List[Set[str]] = []
        for pred in concept_predicates:
            docs = self._posting_lists.get(pred, [])
            sets.append(set(docs))

        if not sets:
            return []

        intersection = sets[0]
        for s in sets[1:]:
            intersection &= s

        return list(intersection)


# ---------- Internal helpers for concept generation ----------

def _parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; attempt simple salvage if extra text appears."""
    if not isinstance(raw, str):
        return []

    raw = raw.strip()
    if not raw:
        return []

    # Normal JSON parse
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

    def assign_concepts(
        self,
        docs: Iterable[Mapping[str, Any]],
        concept_vocabulary: List[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        """
        Returns a mapping:
            doc_id -> { "concept:DimensionName": "Value" }
        
        Example:
            "doc123" -> {
                "concept:film genre": "horror", 
                "concept:release period": "1990s"
            }
        """
        pass


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
      - enrich_documents: assign concepts, update metadata, build inverted concept index
    """

    document_catalog: BaseDocumentCatalog
    vector_index: BaseVectorIndex
    keyword_index: BaseKeywordIndex
    metadata_store: TableMetadataStore
    concept_index: InvertedConceptIndex
    concept_generator: BaseConceptGenerator

    @classmethod
    def from_config(cls, config: Config) -> "IndexManagementPipeline":
        # Single Chroma-backed object used as both catalog and vector index
        chroma_index = ChromaVectorIndex(config)
        keyword = ChromaKeywordIndex(config)
        metadata = TableMetadataStore(config)
        concept_gen = LLMConceptGenerator(config)
        concept_idx = InvertedConceptIndex(config, metadata_store=metadata)

        return cls(
            document_catalog=chroma_index,
            vector_index=chroma_index,
            keyword_index=keyword,
            metadata_store=metadata,
            concept_index=concept_idx,
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

    def add_documents(self, docs: Iterable[Mapping[str, Any]]) -> None:
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
          - TableMetadataStore has a row for each id with uniform schema.
        """
        docs_list = list(docs)
        if not docs_list:
            return

        logger.info(f"IndexManagementPipeline: adding {len(docs_list)} documents.")

        # 1) Source of truth for text + embeddings + raw metadata: Chroma
        self.document_catalog.add_documents(docs_list)

        # 2) Structured metadata table (keeps fields uniform across docs)
        for d in docs_list:
            doc_id = str(d.get("id") or d.get("doc_id"))
            if not doc_id:
                continue
            base_meta = dict(d.get("metadata", {}))
            self.metadata_store.upsert_metadata(doc_id, base_meta)

        # 3) Keyword index (if implemented)
        self.keyword_index.add_documents(docs_list)

    def enrich_documents(self, doc_ids: Sequence[str]) -> None:
        """
        Run concept assignment and update metadata for EXISTING documents.

        After this call:
          - concept:* fields are added to TableMetadataStore (same schema for all docs).
          - InvertedConceptIndex is rebuilt over those concept fields.
        """
        if not doc_ids:
            return

        logger.info(f"IndexManagementPipeline: enriching {len(doc_ids)} documents.")
        # Fetch the actual chunk texts from Chroma (if needed by the concept generator)
        docs = self.document_catalog.get_documents(doc_ids)
        if not docs:
            return

        # Run concept assignment (produces a dict: doc_id -> { "concept:*": value, ... })
        assignments = self.concept_generator.assign_concepts(docs)

        # Update structured metadata store with concept fields
        for doc_id, concept_attrs in assignments.items():
            self.metadata_store.upsert_metadata(doc_id, concept_attrs)

        # Rebuild the inverted concept index
        self.concept_index.materialize_concepts()

    # Convenience getters (unchanged in spirit)
    def get_vector_index(self) -> BaseVectorIndex:
        return self.vector_index

    def get_keyword_index(self) -> BaseKeywordIndex:
        return self.keyword_index

    def get_concept_index(self) -> BaseConceptIndex:
        return self.concept_index

    def get_metadata_store(self) -> BaseMetadataStore:
        return self.metadata_store

    def get_index_manager(self) -> BaseIndexManager:
        return self
