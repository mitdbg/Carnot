from __future__ import annotations
import json
import re
import os
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import hdbscan
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
import dspy
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("concept_generation_outputs")


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Case-insensitive dedupe preserving original order."""
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


class ClusterCentroidSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="A list of mid-granularity concept strings that belong to ONE cluster."
    )
    centroid = dspy.OutputField(
        desc=(
            "Return EXACTLY ONE short noun phrase of the form '<subject> <facet>'. "
            "The subject is a generic singular noun. "
            "The facet is the dominant *attribute type* expressed by the cluster "
            "(e.g., location, nationality, habitat, genre, tear). "
            "Do NOT output specific entities/places/years. "
            "Do NOT output a bare topic like 'fish' or 'criminal films'. "
            "Do NOT combine facets (no 'and', no commas)."
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
                centroid="bird location",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Horror films",
                    "Historical films",
                    "Films set in the future",
                    "Black-and-white films",
                ],
                centroid="film genre",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "1990s films",
                    "Early 1960s films",
                    "Late 1990s films",
                ],
                centroid="film decade",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "2002 films",
                    "films released in 2024",
                    "films shot in 1999",
                ],
                centroid="film year",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=[
                    "Flowers of the Crozet Islands",
                    "Trees of the Marshall Islands",
                    "Trees of the Line Islands",
                ],
                centroid="plant location",
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        """Run the LLM and return a centroid label."""
        concepts_str = "\n".join(f"- {c}" for c in concepts)
        result = self._predict(concepts=concepts_str, demos=self._few_shot_examples)
        centroid = (getattr(result, "centroid", "") or "").strip()
        return centroid


class LLMConceptGenerator:
    """
    Concept generation API:
    - TWO-STAGE: per-query concepts → clustering → centroid labels
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.embedding_model_name = getattr(config, "concept_embedding_model", "all-MiniLM-L6-v2")
        # GMM parameters
        self.n_clusters = getattr(config, "n_clusters", 30)
        # HDBSCAN parameters (unused when using GMM)
        self.min_cluster_size = getattr(config, "min_cluster_size", 3)

        self._centroid_model = ClusterCentroidModel()
        self._concept_vocabulary: List[str] = []

    def fit(
        self,
        concepts: Iterable[str],
        *,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        concepts_list = list(concepts)
        logger.info(f"LLMConceptGenerator: fitting on {len(concepts_list)} concepts.")

        if not concepts_list:
            self._concept_vocabulary = []
            return

        concepts, artifacts = self._fit_two_stage(concepts_list)

        self._concept_vocabulary = concepts
        logger.info(f"LLMConceptGenerator: learned {len(concepts)} concepts.")

        if output_dir is None:
            output_dir = (
                getattr(self.config, "concept_generation_output_dir", None)
                or DEFAULT_OUTPUT_DIR
            )
        self._persist(output_dir=output_dir, artifacts=artifacts)

    def _persist(self, *, output_dir: str | Path, artifacts: Mapping[str, Any]) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "concept_generation_artifacts.json"
        try:
            out_path.write_text(json.dumps(artifacts, indent=2, sort_keys=False))
        except Exception as e:
            logger.warning(f"Failed to write concept generation artifacts to {out_path}: {e}")

    def _fit_two_stage(self, concepts: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """
        TWO-STAGE STRATEGY:
        1. Concepts are embedded and clustered via GMM.
        2. LLM generates a centroid (final concept) for each cluster.
        """
        artifacts: Dict[str, Any] = {
            "mode": "two_stage",
            "embedding_model_name": self.embedding_model_name,
            "n_clusters": self.n_clusters,
        }

        all_concepts = _dedupe_preserve_order(concepts)
        logger.info(f"LLMConceptGenerator: {len(all_concepts)} unique concepts after deduplication.")
        if not all_concepts:
            artifacts["intermediate_concepts"] = []
            artifacts["clusters"] = {}
            return [], artifacts

        model = SentenceTransformer(self.embedding_model_name)
        embeddings = model.encode(all_concepts, show_progress_bar=False)

        logger.info(f"LLMConceptGenerator: clustering with GMM (n_clusters={self.n_clusters}).")
        # HDBSCAN (commented out):
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        # labels = clusterer.fit_predict(embeddings)
        
        # GMM clustering
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        labels = gmm.fit_predict(embeddings)

        clusters: Mapping[int, List[str]] = {}
        for concept, label in zip(all_concepts, labels):
            clusters.setdefault(int(label), []).append(concept)

        cluster_ids = sorted(clusters.keys())
        
        # Store clustering results
        artifacts["intermediate_concepts"] = all_concepts
        artifacts["clusters"] = {str(k): clusters[k] for k in cluster_ids}
        
        # Compute cluster statistics for evaluation
        n_clusters = len([c for c in cluster_ids if c != -1])
        noise_count = len(clusters.get(-1, []))
        cluster_sizes = [len(clusters[c]) for c in cluster_ids if c != -1]
        
        artifacts["cluster_stats"] = {
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_percentage": round(100 * noise_count / len(all_concepts), 2) if all_concepts else 0,
            "cluster_sizes": {
                "min": min(cluster_sizes) if cluster_sizes else 0,
                "max": max(cluster_sizes) if cluster_sizes else 0,
                "mean": round(sum(cluster_sizes) / len(cluster_sizes), 2) if cluster_sizes else 0,
            }
        }
        
        # ============================================================
        # CENTROID GENERATION (COMMENTED OUT FOR CLUSTERING EVALUATION)
        # ============================================================
        cluster_centroids: Dict[str, str] = {}
        centroids_in_cluster_order: List[str] = []
        for cluster_id in tqdm(cluster_ids, desc="two_stage centroids"):
            members = clusters[cluster_id]
            if not members:
                continue
            centroid = self._centroid_model(members)
            if centroid:
                cluster_centroids[str(cluster_id)] = centroid
                centroids_in_cluster_order.append(centroid)
        
        final_concepts = _dedupe_preserve_order(centroids_in_cluster_order)
        artifacts["cluster_centroids"] = cluster_centroids
        artifacts["final_concepts"] = final_concepts
        return final_concepts, artifacts
        # ============================================================
        
        return [], artifacts


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    here = Path(__file__).resolve().parent
    per_query_concepts_path = here / "tmp/concept_generation_artifacts.json"
    output_dir = here / "tmp/clustering_results"

    print(f"Loading concepts from {per_query_concepts_path}")
    with open(per_query_concepts_path, "r") as f:
        data = json.load(f)
    
    # Extract all concepts from per_query_concepts
    all_concepts_raw = []
    for concepts in data["per_query_concepts"].values():
        all_concepts_raw.extend(concepts)
    
    print(f"Extracted {len(all_concepts_raw)} concepts from {len(data['per_query_concepts'])} queries")
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    class _Cfg:
        pass
    
    cfg = _Cfg()
    cfg.min_cluster_size = 20  # HDBSCAN parameter: minimum cluster size
    cfg.n_clusters = 30
    cfg.concept_embedding_model = "all-MiniLM-L6-v2"

    print("Running clustering (centroid generation is disabled)...")
    generator = LLMConceptGenerator(cfg)
    generator.fit(all_concepts_raw, output_dir=output_dir)
    
    # Load and display results
    artifacts_path = output_dir / "concept_generation_artifacts.json"
    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)
    
    stats = artifacts.get("cluster_stats", {})
    clusters = artifacts.get("clusters", {})
    
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)
    print(f"Total unique concepts: {len(artifacts.get('intermediate_concepts', []))}")
    print(f"Number of clusters (excluding noise): {stats.get('n_clusters', 0)}")
    print(f"Noise points (cluster -1): {stats.get('noise_count', 0)} ({stats.get('noise_percentage', 0)}%)")
    print(f"Cluster sizes: min={stats.get('cluster_sizes', {}).get('min', 0)}, "
          f"max={stats.get('cluster_sizes', {}).get('max', 0)}, "
          f"mean={stats.get('cluster_sizes', {}).get('mean', 0)}")
    
    # Print sample clusters
    print("\n" + "=" * 60)
    print("SAMPLE CLUSTERS (first 5 non-noise clusters)")
    print("=" * 60)
    sample_count = 0
    for cluster_id in sorted(clusters.keys(), key=lambda x: int(x)):
        if cluster_id == "-1":
            continue
        if sample_count >= 5:
            break
        members = clusters[cluster_id]
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        for m in members[:8]:
            print(f"  - {m}")
        if len(members) > 8:
            print(f"  ... and {len(members) - 8} more")
        sample_count += 1
    
    print(f"\n✅ Full results saved to: {artifacts_path}")
