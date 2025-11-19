# src/ILCI/eval/eval.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple, Optional

import numpy as np


# ----------------------------- IO helpers ------------------------------------

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------- Retrieval Evaluator --------------------------------

@dataclass
class QueryExample:
    query: str
    gold_docs: Set[str]                # set of relevant doc ids
    pred_docs_ranked: List[str]        # ranked list of predicted doc ids


@dataclass
class RetrievalEvaluator:
    """
    Add query-wise gold + predictions, then compute classic IR metrics.

    Expected per-query data:
      - gold_docs: a SET of relevant doc ids
      - pred_docs_ranked: a ranked LIST of predicted doc ids (highest rank first)
    """
    examples: List[QueryExample] = field(default_factory=list)

    # ---------- Ingestion ----------
    def add(self, query: str, gold_docs: Sequence[str], pred_docs_ranked: Sequence[str]) -> None:
        self.examples.append(
            QueryExample(
                query=query,
                gold_docs=set(gold_docs or []),
                pred_docs_ranked=list(pred_docs_ranked or []),
            )
        )

    # ---------- Per-query (helpers) ----------
    @staticmethod
    def _precision_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        topk = pred[:k]
        if not topk:
            return 0.0
        hit = sum(1 for d in topk if d in gold)
        return hit / float(len(topk))

    @staticmethod
    def _recall_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
        if not gold:
            # Common convention: if no relevant docs, define recall as 0.0 for that query.
            return 0.0
        topk = pred[:k]
        hit = sum(1 for d in topk if d in gold)
        return hit / float(len(gold))

    @staticmethod
    def _average_precision_at_k(pred: Sequence[str], gold: Set[str], k: Optional[int] = None) -> float:
        """
        AP = average of precision@i over positions i where pred[i] is relevant.
        By default uses full list; if k is provided, compute AP@k (truncate).
        If no relevant docs exist, returns 0.0.
        """
        if not gold:
            return 0.0
        if k is None:
            k = len(pred)
        score = 0.0
        hit = 0
        for i, d in enumerate(pred[:k], start=1):  # ranks are 1-based in definitions
            if d in gold:
                hit += 1
                score += hit / float(i)
        return score / float(len(gold))

    @staticmethod
    def _reciprocal_rank(pred: Sequence[str], gold: Set[str]) -> float:
        """
        RR = 1 / rank of first relevant; 0 if none. MRR is mean of RR. 
        """
        for i, d in enumerate(pred, start=1):
            if d in gold:
                return 1.0 / float(i)
        return 0.0

    @staticmethod
    def _dcg_at_k(gains: Sequence[float], k: int) -> float:
        """
        DCG@k = sum_{i=1..k} (gain_i / log2(i+1))
        (using log2(i+1), which is the common formulation).
        """
        k = min(k, len(gains))
        dcg = 0.0
        for i in range(1, k + 1):
            dcg += gains[i - 1] / math.log2(i + 1)
        return dcg

    @staticmethod
    def _ndcg_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
        """
        Binary relevance NDCG@k: gain = 1 for relevant, 0 otherwise.
        """
        if k <= 0:
            return 0.0
        # gains for system ranking
        gains = [1.0 if d in gold else 0.0 for d in pred[:k]]
        dcg = RetrievalEvaluator._dcg_at_k(gains, k)

        # ideal gains (all relevant first)
        ideal_rel = min(len(gold), k)
        ideal_gains = [1.0] * ideal_rel + [0.0] * (k - ideal_rel)
        idcg = RetrievalEvaluator._dcg_at_k(ideal_gains, k)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _hit_rate_at_k(pred: Sequence[str], gold: Set[str], k: int) -> float:
        """
        HitRate@k: 1 if any relevant doc appears in top-k, else 0.
        """
        topk = set(pred[:k])
        return 1.0 if any(d in topk for d in gold) else 0.0

    @staticmethod
    def _r_precision(pred: Sequence[str], gold: Set[str]) -> float:
        """
        R-Precision: Precision@R where R = # of relevant docs for this query.
        """
        R = len(gold)
        if R == 0:
            return 0.0
        return RetrievalEvaluator._precision_at_k(pred, gold, R)

    # ---------- Aggregation ----------
    def evaluate(
        self,
        ks: Sequence[int] = (20, 50, 100),
        include_map: bool = True,
        include_mrr: bool = True,
        include_rprec: bool = True,
        include_hitrate: bool = True,
        ap_at_k: Optional[int] = None,   # if set, report MAP@K in addition to MAP (full)
        ndcg: Sequence[int] = (10, 20, 50, 100),
    ) -> Dict:
        """
        Returns a metrics dict with per-K and global scores (means across queries).
        """
        # Sanitize and sort Ks
        Ks = sorted(set(int(k) for k in ks if k > 0))
        NDCGs = sorted(set(int(k) for k in ndcg if k > 0))

        # holders
        prec_at: Dict[int, List[float]] = {k: [] for k in Ks}
        rec_at: Dict[int, List[float]] = {k: [] for k in Ks}
        hit_at: Dict[int, List[float]] = {k: [] for k in Ks} if include_hitrate else {}
        ndcg_at: Dict[int, List[float]] = {k: [] for k in NDCGs}

        ap_list: List[float] = []
        ap_atk_list: List[float] = [] if ap_at_k else None
        rr_list: List[float] = [] if include_mrr else None
        rprec_list: List[float] = [] if include_rprec else None

        for ex in self.examples:
            pred = ex.pred_docs_ranked
            gold = ex.gold_docs

            # P@K, R@K, Hit@K
            for k in Ks:
                prec_at[k].append(self._precision_at_k(pred, gold, k))
                rec_at[k].append(self._recall_at_k(pred, gold, k))
                if include_hitrate:
                    hit_at[k].append(self._hit_rate_at_k(pred, gold, k))

            # NDCG@K
            for k in NDCGs:
                ndcg_at[k].append(self._ndcg_at_k(pred, gold, k))

            # MAP (+ optionally MAP@K)
            if include_map:
                ap_list.append(self._average_precision_at_k(pred, gold, None))
                if ap_at_k:
                    ap_atk_list.append(self._average_precision_at_k(pred, gold, ap_at_k))  # type: ignore

            # MRR
            if include_mrr:
                rr_list.append(self._reciprocal_rank(pred, gold))  # type: ignore

            # R-Precision
            if include_rprec:
                rprec_list.append(self._r_precision(pred, gold))  # type: ignore

        # Aggregate means
        def mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        out = {
            "num_queries": len(self.examples),
            "precision": {k: mean(prec_at[k]) for k in Ks},
            "recall":    {k: mean(rec_at[k]) for k in Ks},
            "ndcg":      {k: mean(ndcg_at[k]) for k in NDCGs},
        }
        if include_hitrate:
            out["hit_rate"] = {k: mean(hit_at[k]) for k in Ks}
        if include_map:
            out["map"] = mean(ap_list)
            if ap_atk_list is not None:
                out[f"map@{ap_at_k}"] = mean(ap_atk_list)
        if include_mrr:
            out["mrr"] = mean(rr_list or [])
        if include_rprec:
            out["r_precision"] = mean(rprec_list or [])

        return out

    # ---------- Convenience: load and evaluate JSONL like your scripts ----------
    @classmethod
    def from_jsonl_files(
        cls,
        gold_path: str,
        pred_path: str,
        gold_key_query: str = "query",
        gold_key_docs: str = "docs",
        pred_key_query: str = "query",
        pred_key_docs: str = "docs",
    ) -> "RetrievalEvaluator":
        """
        Build an evaluator from:
          gold jsonl: {"query": ..., "docs": [gold_doc_ids]}
          pred jsonl: {"query": ..., "docs": [pred_doc_ids_ranked]}
        """
        gold = list(read_jsonl(gold_path))
        pred = list(read_jsonl(pred_path))
        pred_map = {row[pred_key_query]: row[pred_key_docs] for row in pred}

        ev = cls()
        for row in gold:
            q = row[gold_key_query]
            gdocs = row.get(gold_key_docs, []) or []
            pdocs = pred_map.get(q, []) or []
            ev.add(q, gdocs, pdocs)
        return ev


# ----------------------------- Accuracy Evaluator --------------------------------

class AccuracyEvaluator:
    def __init__(self):
        self.em_scores = []
        self.f1_scores = []

    def normalize_answer(self, s: str) -> str:
        """Lowercase, remove punctuation, and extra whitespace."""
        return ' '.join(re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()).split())

    def exact_match(self, prediction: str, ground_truth: str) -> int:
        """Compute exact match score."""
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score."""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / len(prediction_tokens)
        recall = num_common / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate EM and F1 scores."""
        assert len(predictions) == len(ground_truths), "Mismatch between predictions and ground truths."
        for pred, gt in zip(predictions, ground_truths):
            self.em_scores.append(self.exact_match(pred, gt))
            self.f1_scores.append(self.f1_score(pred, gt))
        return {
            "exact_match": np.mean(self.em_scores),
            "f1_score": np.mean(self.f1_scores)
        }

