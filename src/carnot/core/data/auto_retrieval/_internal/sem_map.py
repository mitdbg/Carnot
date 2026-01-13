from __future__ import annotations

import json
import logging
import sys
import os
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union, get_args, get_origin

from pathlib import Path
import dspy
import palimpzest as pz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
            "  * CONSTRAINT: Do not end the name with a colon.\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[str]'.\n"
            "  * CONSTRAINT: Only use 'List[str]' for lists. Do NOT use 'List[int]' or 'List[float]'.\n"
            "  * Determine the concept's cardinality. If the concept tends to have multiple mentions per document, use 'List[str]'.\n"
            "    If it implies a single value, use 'str', 'int', or 'float'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "  * CRITICAL: In the description, explicitly state: 'If the document does not mention [concept], set this field to null'. Do not use 'None'.\n"
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
            "- name: Create a hierarchy. Base hierarchy must preserve word order and use EXACT words from input. Allow expansion (':subtype') ONLY when essential.\n"
            "  * CONSTRAINT: Do not end the name with a colon.\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[str]'.\n"
            "  * CONSTRAINT: Only use 'List[str]' for lists. Do NOT use 'List[int]' or 'List[float]'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "  * CRITICAL: In the description, explicitly state: 'If the document does not mention [concept], set this field to null'. Do not use 'None'.\n"
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
                    {"name": "film:location:city", "type": "List[str]", "desc": "Distinct film cities mentioned. If the document does not mention any cities, set this field to null."},
                    {"name": "film:location:state", "type": "List[str]", "desc": "Distinct film states mentioned. If the document does not mention any states, set this field to null."},
                    {"name": "film:location:province", "type": "List[str]", "desc": "Distinct film provinces mentioned. If the document does not mention any provinces, set this field to null."},
                    {"name": "film:location:country", "type": "List[str]", "desc": "Distinct film countries mentioned. If the document does not mention any countries, set this field to null."},
                    {"name": "film:location:continent", "type": "List[str]", "desc": "Distinct film continents mentioned. If the document does not mention any continents, set this field to null."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to null."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["film decade"],
                concept_schema=json.dumps([
                    {"name": "film:decade", "type": "int", "desc": "Decade the film was released. If the document does not mention a film decade, set this field to null."},
                ])
            ).with_inputs("concepts"),
            # ... keep other examples, just ensure 'None' -> 'null' in descriptions
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
                    {"name": "film:location", "type": "List[str]", "desc": "Distinct film locations mentioned. If the document does not mention any film locations, set this field to null."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to null."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["bird color"],
                concept_schema=json.dumps([
                    {"name": "bird:color", "type": "List[str]", "desc": "Distinct bird colors mentioned. If the document does not mention any bird colors, set this field to null."},
                ])
            ).with_inputs("concepts")
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        return result.concept_schema

def _type_from_str(t: str) -> Any:
    """Parse type string to Python type. Only allows str, int, float, List[str]."""
    t_clean = str(t).strip()
    if t_clean == "int": return int
    if t_clean == "float": return float
    if t_clean == "str": return str
    if t_clean == "List[str]": return List[str]
    raise ValueError(f"Unsupported schema type: {t_clean!r}")

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
    
    logger.info(f"Running sem_map with {len(rows)} rows...")
    logger.info(f"Number of concept schema columns: {len(concept_schema_cols)}")

    dataset = pz.MemoryDataset(id="sem-map", vals=rows)
    cols_for_pz = [dict(c) for c in concept_schema_cols]
    dataset = dataset.sem_map(cols=cols_for_pz)
    config = pz.QueryProcessorConfig(available_models=[pz.Model.GPT_5])
    output = dataset.run(config=config, max_quality=True)
    
    # Log token usage if available from palimpzest execution stats
    exec_stats = getattr(output, "execution_stats", None)
    if exec_stats is not None:
        try:
            total_in = getattr(exec_stats, "total_input_tokens", None)
            total_out = getattr(exec_stats, "total_output_tokens", None)
            total_tokens = getattr(exec_stats, "total_tokens", None)
            num_rows = len(rows) if rows is not None else 0
            avg_out_per_row = (float(total_out) / float(num_rows)) if (total_out is not None and num_rows > 0) else None
            
            if total_in is not None and total_out is not None and total_tokens is not None:
                logger.info(f"sem_map token usage — input: {total_in:,} | output: {total_out:,} | total: {total_tokens:,}")
            if avg_out_per_row is not None:
                logger.info(f"sem_map average output tokens per row: {avg_out_per_row:,.2f}")
        except Exception:
            pass
    
    col_names = [c["name"] for c in concept_schema_cols]

    # Ensure every input doc_id exists even if execution drops a record
    results: Dict[str, Dict[str, Any]] = {r["id"]: {} for r in rows}

    for rec in getattr(output, "data_records", []):
        doc_id = str(getattr(rec, "id", "")).strip()
        if not doc_id:
            continue
        if doc_id not in results:
            results[doc_id] = {}

        out: Dict[str, Any] = {}
        for c in col_names:
            v = getattr(rec, c, None)
            v = _dedupe_list(v)
            if v is None or v == "" or v == []:
                continue
            out[c] = v

        results[doc_id] = out

    return results, concept_schema_cols


def _is_list_of_str_type(tp: Any) -> bool:
    """Check if type is List[str]."""
    origin = get_origin(tp)
    if origin is not list:
        return False
    args = get_args(tp)
    return len(args) == 1 and args[0] is str


def _is_taggable_type(tp: Any) -> bool:
    """Taggable = str or List[str] ONLY."""
    return tp is str or _is_list_of_str_type(tp)


def _canon_tag_suffix(v: Any) -> str:
    """Canonical string suffix used in the tag key (keeps spaces, trims/normalizes)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return " ".join(v.strip().split())
    # If model accidentally returns numbers for a taggable col, still stringify
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v).strip()
    return str(v).strip()


def _normalize_tag_values(v: Any) -> List[str]:
    """
    Normalize a tag value, potentially splitting comma-separated lists.
    
    Smart splitting rules:
    - Split on ", " (comma followed by space) to handle lists like "Asia, Africa"
    - This preserves hyphenated compound terms like "romantic comedy-drama"
    - Normalize to lowercase for consistent tag matching
    """
    sfx = _canon_tag_suffix(v)
    if not sfx:
        return []
    
    # Split on ", " (comma followed by space)
    parts = [p.strip() for p in sfx.split(", ")]
    # Normalize to lowercase for consistent tag matching
    return [p.lower() for p in parts if p]


def _cast_scalar(v: Any, tp: Any) -> Any:
    """Best-effort cast into int/float/bool. Return None if not castable."""
    if v is None:
        return None

    # If LLM incorrectly returns list for scalar, take first non-null
    if isinstance(v, list):
        v = next((x for x in v if x is not None and x != ""), None)
        if v is None:
            return None

    if tp is int:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return int(float(s))  # handles "1999.0"
            except Exception:
                return None
        return None

    if tp is float:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except Exception:
                return None
        return None

    if tp is bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().casefold()
            if s in {"true", "t", "yes", "y", "1"}:
                return True
            if s in {"false", "f", "no", "n", "0"}:
                return False
        return None

    return None


def expand_sem_map_results_to_tags(
    results: Dict[str, Dict[str, Any]],
    schema: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Tagification Strategy:
    
    1. Taggable columns (str or List[str] ONLY):
       - Convert each unique value to a boolean tag column (e.g., 'film:genre:action', 'film:genre:comedy')
       - ALL documents MUST have ALL tag columns in their metadata
       - Documents that originally had that value → set tag to True
       - Documents that didn't have that value → set tag to False
       
    2. Non-taggable scalar columns (int, float, bool ONLY):
       - Keep as-is (no tagification)
       - Only add to metadata if the document has a value
       - If document doesn't have a value, don't include the key at all (sparse representation)
    
    3. All other types are dropped (no str, no lists in final output).
    
    Returns:
      - expanded_results: doc_id -> { tag_col: bool, scalar_col: int|float|bool }
        * Every document has ALL tag columns (True or False)
        * Scalar columns only present if document has a value
        * Output values are ONLY int/float/bool (no str, no lists)
      - expanded_schema: schema where taggable cols are replaced by bool tag columns;
                        scalar cols kept iff int/float/bool.
      - expanded_stats: per-column selectivity stats (present/total/selectivity)
        * For bool tags: present = count of True
        * For scalar int/float: present = count of docs that have the key (sparse)
    """
    type_by_name: Dict[str, Any] = {c["name"]: c["type"] for c in schema}

    taggable_cols = {n for n, tp in type_by_name.items() if _is_taggable_type(tp)}
    scalar_cols = {n for n, tp in type_by_name.items() if tp in (int, float, bool)}

    # 1) Collect universe of tag values per taggable column
    tag_values: Dict[str, set[str]] = {c: set() for c in taggable_cols}
    for _, cols in results.items():
        for col, v in cols.items():
            if col not in taggable_cols:
                continue
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                # Normalize and potentially split comma-separated values
                for normalized in _normalize_tag_values(x):
                    tag_values[col].add(normalized)

    # 2) Build expanded schema:
    #    - taggable => many bool tag columns
    #    - scalar int/float/bool => keep
    #    - everything else => dropped
    expanded_schema: List[Dict[str, Any]] = []
    for col_def in schema:
        name = col_def["name"]
        tp = col_def["type"]

        if name in taggable_cols:
            for sfx in sorted(tag_values[name], key=lambda s: s.casefold()):
                expanded_schema.append(
                    {
                        "name": f"{name}:{sfx}",
                        "type": bool,
                        "desc": f"True if {sfx!r} appears in {name}.",
                    }
                )
            continue

        if tp in (int, float, bool):
            expanded_schema.append(col_def)
            continue

        # Everything else is dropped to satisfy "only int/float/bool values" constraint.
        logger.warning(f"Dropping non-supported metadata column {name!r} with type {tp!r}")

    all_bool_cols = [c["name"] for c in expanded_schema if c["type"] is bool]
    scalar_schema_cols = [c["name"] for c in expanded_schema if c["type"] in (int, float)]

    total = float(len(results))

    # Count "present" occurrences inline while building expanded_results.
    # - For bool tags: present means True
    # - For scalar cols: present means key exists (sparse)
    present: Dict[str, float] = {name: 0.0 for name in (all_bool_cols + scalar_schema_cols)}

    # 3) Build expanded results
    expanded_results: Dict[str, Dict[str, Any]] = {}
    for doc_id, cols in results.items():
        # Dense bool grid for tags
        out: Dict[str, Any] = {k: False for k in all_bool_cols}

        for col, v in cols.items():
            # taggable => set tag columns True
            if col in taggable_cols:
                vs = v if isinstance(v, list) else [v]
                for x in vs:
                    # Normalize and potentially split comma-separated values
                    for normalized in _normalize_tag_values(x):
                        key = f"{col}:{normalized}"
                        # Key should exist if it was in the global universe; guard anyway
                        if key in out and out[key] is False:
                            out[key] = True
                            present[key] += 1.0
                continue

            # scalar => keep sparse, cast to int/float/bool
            if col in scalar_cols:
                tp = type_by_name[col]
                casted = _cast_scalar(v, tp)
                if casted is not None:
                    out[col] = casted
                    present[col] += 1.0
                continue

            # drop everything else

        expanded_results[doc_id] = out

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