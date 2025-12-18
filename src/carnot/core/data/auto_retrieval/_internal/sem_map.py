from __future__ import annotations

import json
from dataclasses import dataclass
import os
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union, get_args, get_origin

from pathlib import Path
import dspy
import palimpzest as pz

class SemMapStrategy(str, Enum):
    HIERARCHY_FIRST = "hierarchy_first"
    FLAT = "flat"

class FlatConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="List of concept phrases. Normalize them into a FLAT, canonical schema."
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return a JSON array of objects with keys: name, type, desc.\n"
            "Format rules:\n"
            "- name: Create a colon-delimited hierarchy (e.g., 'film genre' -> 'film:genre').\n"
            "  * CONSTRAINT: Use EXACT words from the input. Do not introduce new words.\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[...]'.\n"
            "  * Determine the concept's cardinality. If the concept tends to have multiple mentions per document, use 'List[...]'.\n"
            "    If it implies a single value, use 'str', 'int', or 'float'.\n"
            "    Example: locations often have multiple mentions -> 'List[str]'; year is usually a single value -> 'int'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "Output must be valid JSON only. No markdown."
        )
    )


class HierarchyFirstConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="List of concept phrases. Normalize them into a HIERARCHY-FIRST schema."
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return a JSON array of objects with keys: name, type, desc.\n"
            "Format rules:\n"
            "- name: Create a hierarchy. Base hierarchy must preserve the same word order as the input and use EXACT words from the input. Allow expansion (':subtype') ONLY when essential and necessary for granularity.\n"
            "  * Example: 'film location' -> 'film:location:city', 'film:location:state', 'film:location:province', 'film:location:country', 'film:location:continent'\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[...]'.\n"
            "  * Determine the concept's cardinality. If the concept tends to have multiple mentions per document, use 'List[...]'.\n"
            "    If it implies a single value, use 'str', 'int', or 'float'.\n"
            "    Example: locations often have multiple mentions -> 'List[str]'; year is usually a single value -> 'int'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "Output must be valid JSON only. No markdown."
        )
    )

class HierarchyFirstConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(HierarchyFirstConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location"],
                concept_schema=json.dumps([
                    {"name": "film:location:city", "type": "List[str]", "desc": "Distinct film cities mentioned. If the document does not mention any cities, set this field to None."},
                    {"name": "film:location:state", "type": "List[str]", "desc": "Distinct film states mentioned. If the document does not mention any states, set this field to None."},
                    {"name": "film:location:province", "type": "List[str]", "desc": "Distinct film provinces mentioned. If the document does not mention any provinces, set this field to None."},
                    {"name": "film:location:country", "type": "List[str]", "desc": "Distinct film countries mentioned. If the document does not mention any countries, set this field to None."},
                    {"name": "film:location:continent", "type": "List[str]", "desc": "Distinct film continents mentioned. If the document does not mention any continents, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["film decade"],
                concept_schema=json.dumps([
                    {"name": "film:decade", "type": "int", "desc": "Decade the film was released. If the document does not mention a film decade, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["person name"],
                concept_schema=json.dumps([
                    {"name": "person:name:first", "type": "str", "desc": "First name of the person. If the document does not mention a person's first name, set this field to None."},
                    {"name": "person:name:last", "type": "str", "desc": "Last name of the person. If the document does not mention a person's last name, set this field to None."},
                ])
            ).with_inputs("concepts")
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        return result.concept_schema

class FlatConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(FlatConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location"],
                concept_schema=json.dumps([
                    {"name": "film:location", "type": "List[str]", "desc": "Distinct film locations mentioned. If the document does not mention any film locations, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["bird color"],
                concept_schema=json.dumps([
                    {"name": "bird:color", "type": "List[str]", "desc": "Distinct bird colors mentioned. If the document does not mention any bird colors, set this field to None."},
                ])
            ).with_inputs("concepts")
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        return result.concept_schema

@dataclass(frozen=True)
class PruneSemMapResults:
    # TODO: start simple, evolve this into a utility score combining selectivity and query frequency.
    pass


def _type_from_str(t: str) -> Any:
    # Simplified parser to map clean strings to types
    t_clean = str(t).strip()
    if t_clean == "int": return int
    if t_clean == "float": return float
    if t_clean == "str": return str
    if t_clean == "List[str]": return List[str]
    if t_clean == "List[int]": return List[int]
    if t_clean == "List[float]": return List[float]
    return List[str] # Default fallback


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
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    strat = SemMapStrategy(strategy)
    model: dspy.Module = FlatConceptSchemaModel() if strat is SemMapStrategy.FLAT else HierarchyFirstConceptSchemaModel()
    
    concept_schema_cols: List[Dict[str, Any]] = []

    for concept in concepts:
        raw_output = model([concept])
        
        if not isinstance(raw_output, str):
            raise TypeError(f"DSPy concept_schema output expected string, got {type(raw_output)}")
        
        try:
            schema_list = json.loads(raw_output)
            # Ensure it's a list
            if not isinstance(schema_list, list):
                 raise ValueError("Output is not a JSON list")
        except Exception as e:
            print(f"DEBUG: Raw='{raw_output}'")
            raise ValueError(f"Failed to parse DSPy concept_schema output as JSON for concept '{concept}'.") from e

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

    if not rows:
        return {}, concept_schema_cols

    dataset = pz.MemoryDataset(id="sem-map", vals=rows)
    cols_for_pz = [dict(c) for c in concept_schema_cols]
    dataset = dataset.sem_map(cols=cols_for_pz)
    output = dataset.run(max_quality=True)
    
    col_names = [c["name"] for c in concept_schema_cols]
    results: Dict[str, Dict[str, Any]] = {}

    for rec in getattr(output, "data_records", []):
        doc_id = str(getattr(rec, "id", "")).strip()
        if not doc_id:
            continue

        out: Dict[str, Any] = {}
        for c in col_names:
            v = getattr(rec, c, None)

            v = _dedupe_list(v)
            if v is None or v == "" or v == []:
                continue
            out[c] = v

        results[doc_id] = out    

    return results, concept_schema_cols


def _is_list_type(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is list

def _is_taggable_type(tp: Any) -> bool:
    return tp is str or _is_list_type(tp)

def _canon_tag_suffix(v: Any) -> str:
    """Canonical string suffix used in the tag key (keeps spaces, trims/normalizes)."""
    if v is None:
        return ""
    if isinstance(v, str):
        s = " ".join(v.strip().split())
        return s
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    return str(v).strip()


def expand_sem_map_results_to_tags(
    results: Dict[str, Dict[str, Any]],
    schema: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Returns:
      - expanded_results: doc_id -> { tag_col: True, scalar_col: value, ... }
      - expanded_schema: schema where taggable cols are replaced by bool tag columns
      - expanded_stats: present/total/selectivity for each expanded column
    """
    type_by_name: Dict[str, Any] = {c["name"]: c["type"] for c in schema}
    taggable_cols = {name for name, tp in type_by_name.items() if _is_taggable_type(tp)}

    tag_values: Dict[str, set[str]] = {c: set() for c in taggable_cols}
    for _, cols in results.items():
        for col, v in cols.items():
            if col not in taggable_cols:
                continue
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                sfx = _canon_tag_suffix(x)
                if sfx:
                    tag_values[col].add(sfx)

    expanded_schema: List[Dict[str, Any]] = []
    for col_def in schema:
        name = col_def["name"]
        if name not in taggable_cols:
            expanded_schema.append(col_def)
            continue

        for sfx in sorted(tag_values[name], key=lambda s: s.casefold()):
            tag_name = f"{name}:{sfx}"
            expanded_schema.append(
                {
                    "name": tag_name,
                    "type": bool,
                    "desc": f"True if {sfx!r} appears in {name}.",
                }
            )

    expanded_results: Dict[str, Dict[str, Any]] = {}
    
    all_bool_cols = [col["name"] for col in expanded_schema if col["type"] is bool]
    
    for doc_id, cols in results.items():
        out: Dict[str, Any] = {k: False for k in all_bool_cols}
        
        for col, v in cols.items():
            if col not in taggable_cols:
                out[col] = v
                continue
                
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                sfx = _canon_tag_suffix(x)
                if sfx:
                    out[f"{col}:{sfx}"] = True
        expanded_results[doc_id] = out

    total = float(len(results))
    present: Dict[str, float] = {c["name"]: 0.0 for c in expanded_schema}
    
    for _, cols in expanded_results.items():
        for k, v in cols.items():
            if k not in present:
                continue
            if isinstance(v, bool):
                if v:
                    present[k] += 1.0
            else:
                if v is not None:
                    present[k] += 1.0

    expanded_stats: Dict[str, Dict[str, float]] = {}
    for k, p in present.items():
        sel = (p / total) if total else 0.0
        expanded_stats[k] = {"present": p, "total": total, "selectivity": sel}

    return expanded_results, expanded_schema, expanded_stats


# Example usage
if __name__ == "__main__":
    def _type_to_str(tp: Any) -> str:
        if tp is str:
            return "str"
        if tp is int:
            return "int"
        if tp is float:
            return "float"
        if tp is bool:
            return "bool"
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is list and len(args) == 1:
            return f"List[{_type_to_str(args[0])}]"
        return str(tp)

    def _load_example_rows(example_path: Path) -> List[Dict[str, str]]:
        lines = example_path.read_text(encoding="utf-8").splitlines()
        rows: List[Dict[str, str]] = []
        for i, line in enumerate(lines, start=1):
            text = " ".join(line.strip().split())
            if not text:
                continue
            rows.append({"id": f"doc_{i:03d}", "text": text})
        return rows

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    here = Path(__file__).resolve().parent
    example_path = here / "example_sem_map.txt"
    data_rows = _load_example_rows(example_path)

    concepts = [
        "amphibian location",
        "film classification",
        "fish classification",
        "book release-year",
        "crime film theme",
    ]
    strategy = SemMapStrategy.HIERARCHY_FIRST

    sem_results, concept_schema_cols = sem_map(concepts=concepts, data=data_rows, strategy=strategy)

    expanded_results, expanded_schema, expanded_stats = expand_sem_map_results_to_tags(
        sem_results, concept_schema_cols
    )

    sem_payload = {
        "strategy": strategy.value,
        "concepts": list(concepts),
        "concept_schema_cols": [
            {"name": c["name"], "type": _type_to_str(c["type"]), "desc": c.get("desc", "")}
            for c in concept_schema_cols
        ],
        "results": sem_results,
    }
    expanded_payload = {
        "schema": [
            {"name": c["name"], "type": _type_to_str(c["type"]), "desc": c.get("desc", "")}
            for c in expanded_schema
        ],
        "results": expanded_results,
        "stats": expanded_stats,
    }

    sem_out_path = here / "sem_map/example_sem_map_output.json"
    expanded_out_path = here / "sem_map/example_sem_map_tagified_output.json"
    
    sem_out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_out_path.parent.mkdir(parents=True, exist_ok=True)

    sem_out_path.write_text(json.dumps(sem_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    expanded_out_path.write_text(
        json.dumps(expanded_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote sem_map output to: {sem_out_path}")
    print(f"Wrote expanded tag output to: {expanded_out_path}")

    raise SystemExit(0)
