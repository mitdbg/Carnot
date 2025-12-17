from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import dspy
import palimpzest as pz

class SemMapStrategy(str, Enum):
    HIERARCHY_FIRST = "hierarchy_first"
    FLAT = "flat"

import dspy
from typing import List

class FlatConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc=(
            "List of concept phrases. Normalize into a FLAT canonical schema.\n"
            "- Mapping Rule: Convert the input phrase structure directly into hierarchy levels.\n"
            "- Strict Token Rule: The output name must consist ONLY of the exact words present in the input phrase. Do not add any words not found in the input.\n"
            "- The depth is determined by the input phrase.\n"
            "- Replace spaces with colons to define levels (e.g., 'A B C' -> 'A:B:C').\n"
            "- Do NOT infer any additional hierarchy/subtypes that are not explicitly in the input phrase.\n"
            "Examples:\n"
            "  'film location' -> name: 'film:location'\n"
            "Counter-example: 'film location' -> name: 'film:location:city' (incorrect, you must not infer any additional hierarchy/subtypes.)"
        )
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of objects: {name,type,desc}.\n"
            "- name: The colon-separated hierarchy path derived strictly from the input.\n"
            "- type: Python value type. JUDGE CARDINALITY per single context/passage:\n"
            "  - Use 'List[...]' if multiple values typically appear (e.g., genres, cast members, locations).\n"
            "  - Use scalar (int/str/float) if only one value is expected (e.g., release year, birth date, ISBN, rating).\n"
            "- desc: Natural language description focusing on the LEAF node of the hierarchy.\n"
            "  Example: name='film:location' -> desc='distinct film locations mentioned.'\n"
            "  Example: name='book:year' -> desc='year the book was released.'"
        )
    )

class HierarchyFirstConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc=(
            "List of concept phrases. Normalize into a HIERARCHY-FIRST schema.\n"
            "- Base Rule: Start by converting the input phrase into a base hierarchy (e.g., 'film location' -> 'film:location').\n"
            "- Base Token Rule: The base hierarchy must consist ONLY of the exact words present in the input phrase.\n"
            "- Inference Rule: You MAY infer at most ONE extra hierarchy level (':subtype') if it provides necessary granularity.\n"
            "- Do NOT output deep chains (4+ levels) unless absolutely necessary.\n"
            "Examples:\n"
            "  'film location' -> Infer subtypes -> 'film:location:city', 'film:location:state', 'film:location:province', 'film:location:country', 'film:location:continent'\n"
            "  'person name' -> Infer subtypes -> 'person:name:first', 'person:name:last'"
        )
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of objects: {name,type,desc}.\n"
            "- name: 'domain:attribute' or 'domain:attribute:subtype'.\n"
            "- type: Python value type. JUDGE CARDINALITY per single context/passage:\n"
            "  - Use 'List[...]' if multiple values typically appear (e.g., genres, cast members).\n"
            "  - Use scalar (int/str/float) if only one value is expected (e.g., release year, birth date).\n"
            "- desc: Natural language description focusing on the LEAF node.\n"
            "  Format: 'distinct <domain> <leaf_plural> mentioned.' (for Lists) OR '<domain> <leaf_singular> mentioned.' (for scalars)\n"
            "  Example: name='film:location:city' -> desc='distinct film cities mentioned.'\n"
            "  CRITICAL: The extracted values must match the semantic granularity of the leaf node.\n"
            "  (e.g., if leaf is ':city', it must only extract cities, not neighborhoods, states, provinces, countries, or continents.)"
        )
    )

class HierarchyFirstConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(HierarchyFirstConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location", "book release_year", "bird color"],
                concept_schema=[
                    {
                        "name": "film:location:city", 
                        "type": "List[str]", 
                        "desc": "distinct film cities mentioned."
                    },
                    {
                        "name": "film:location:country", 
                        "type": "List[str]", 
                        "desc": "distinct film countries mentioned."
                    },
                    {
                        "name": "book:release_year", 
                        "type": "int", 
                        "desc": "year the book was released."
                    },
                    {
                        "name": "bird:color", 
                        "type": "List[str]", 
                        "desc": "distinct bird colors mentioned."
                    },
                ],
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        raw = getattr(result, "concept_schema")
        return raw


class FlatConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(FlatConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location", "film location city", "book release_year"],
                concept_schema=[
                    {
                        "name": "film:location", 
                        "type": "List[str]", 
                        "desc": "distinct film locations mentioned."
                    },
                    {
                        "name": "film:location:city", 
                        "type": "List[str]", 
                        "desc": "distinct film cities mentioned."
                    },
                    {
                        "name": "book:release_year", 
                        "type": "int", 
                        "desc": "year the book was released."
                    },
                ],
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        raw = getattr(result, "concept_schema")
        return raw


@dataclass(frozen=True)
class PruneMetadata:
    min_selectivity: float = 0.01
    max_selectivity: float = 0.99

    # TODO: evolve this into a utility score combining selectivity and query frequency.


def _type_from_str(t: str) -> Any:
    tt = str(t).strip()
    if tt in ("int", "builtins.int"):
        return int
    if tt in ("float", "builtins.float"):
        return float
    if tt in ("str", "builtins.str"):
        return str
    if tt in ("List[str]", "list[str]", "typing.List[str]"):
        return List[str]
    if tt in ("List[int]", "list[int]", "typing.List[int]"):
        return List[int]
    if tt in ("List[float]", "list[float]", "typing.List[float]"):
        return List[float]
    return List[str]


def _dedupe_list(vals: Any) -> Any:
    if not isinstance(vals, list):
        return vals
    out: List[Any] = []
    seen: set[str] = set()
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            s = " ".join(v.strip().split())
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        else:
            k = str(v)
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
    return out


def sem_map(
    *,
    concepts: Sequence[str],
    data: Sequence[Mapping[str, str]],
    strategy: Union[SemMapStrategy, str] = SemMapStrategy.FLAT,
    prune: bool = True,
    prune_config: PruneMetadata = PruneMetadata(),
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    strat = SemMapStrategy(strategy)
    model: dspy.Module = FlatConceptSchemaModel() if strat is SemMapStrategy.FLAT else HierarchyFirstConceptSchemaModel()
    raw_schema = model(list(concepts))
    if not isinstance(raw_schema, str):
        raise TypeError("DSPy concept_schema output must be a JSON string.")
    try:
        schema_list = json.loads(raw_schema)
    except Exception as e:
        raise ValueError("Failed to parse DSPy concept_schema output as JSON.") from e

    concept_schema_cols: List[Dict[str, Any]] = []
    for col in schema_list:
        concept_schema_cols.append(
            {"name": col["name"], "type": _type_from_str(col["type"]), "desc": col.get("desc", "...")}
        )

    rows: List[Dict[str, str]] = []
    for d in data:
        doc_id = str(d.get("id", "")).strip()
        text = str(d.get("text", "")).strip()
        if doc_id and text:
            rows.append({"id": doc_id, "text": text})

    if not rows or not concept_schema_cols:
        return {}, concept_schema_cols, {}

    dataset = pz.MemoryDataset(id="sem-map", vals=rows)
    dataset = dataset.sem_map(cols=concept_schema_cols)
    output = dataset.run(max_quality=True)
    df = output.to_df()

    col_names = [c["name"] for c in concept_schema_cols]
    total = len(rows)
    present: Dict[str, int] = {c: 0 for c in col_names}
    results: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        out: Dict[str, Any] = {}
        for c in col_names:
            v = row.get(c, None)
            v = _dedupe_list(v)
            if v is None or v == "" or v == []:
                continue
            present[c] = present.get(c, 0) + 1
            out[c] = v
        results[doc_id] = out

    stats: Dict[str, Dict[str, float]] = {}
    for c in col_names:
        sel = (float(present.get(c, 0)) / float(total)) if total else 0.0
        stats[c] = {"present": float(present.get(c, 0)), "total": float(total), "selectivity": sel}

    if not prune:
        return results, concept_schema_cols, stats

    keep = {c for c in col_names if prune_config.min_selectivity <= stats[c]["selectivity"] <= prune_config.max_selectivity}
    pruned_schema = [c for c in concept_schema_cols if c["name"] in keep]
    pruned_results: Dict[str, Dict[str, Any]] = {doc_id: {k: v for k, v in cols.items() if k in keep} for doc_id, cols in results.items()}
    return pruned_results, pruned_schema, stats
