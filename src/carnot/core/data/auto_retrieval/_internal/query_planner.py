from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dspy

logger = logging.getLogger(__name__)


class SelectFiltersSignature(dspy.Signature):
    """Given a query and available metadata filters, select which filters are needed."""

    query = dspy.InputField(desc="The user's search query.")
    available_filters = dspy.InputField(
        desc="List of available metadata filters with their types."
    )
    selected_filters = dspy.OutputField(
        desc=(
            "Return a JSON array of filter names that are needed to answer the query. "
            "Return [] if no filters are appropriate."
        )
    )


class GenerateWhereClauseSignature(dspy.Signature):
    """Generate a ChromaDB where clause based on query and filter schema."""

    query = dspy.InputField(desc="The user's search query.")
    filter_schema = dspy.InputField(desc="Selected filters with their types and allowed values.")
    chroma_instructions = dspy.InputField(desc="Instructions for generating valid ChromaDB where clauses.")
    where_clause = dspy.OutputField(desc="Valid ChromaDB where clause as a JSON object.")


def _parse_json(raw: str, default: Any) -> Any:
    """Parse JSON from LLM output, handling markdown code blocks."""
    if not raw:
        return default
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse JSON: {raw[:100]}...")
        return default


CHROMA_WHERE_INSTRUCTIONS = """
ChromaDB Where Clause Rules:

1. BOOLEAN TAGS (type: bool):
   - Tags are stored as flattened keys with true/false values
   - To INCLUDE documents with a tag: {"filter:value": true}
   - To EXCLUDE documents with a tag: {"filter:value": false}
   - Example: To find comedy films, use {"film:genre:comedy": true}
   - Example: To find NON-comedy films, use {"film:genre:comedy": false}
   - For multiple values (OR), use: {"$or": [{"film:genre:comedy": true}, {"film:genre:drama": true}]}

2. NUMERIC FIELDS (type: int/float):
   - Stored directly: "film:year": 1999
   - Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
   - Example: {"film:year": {"$gte": 1990}}
   - Example: {"film:year": {"$in": [1990, 1991, 1992]}}

3. COMBINING FILTERS:
   - Use $and for multiple conditions: {"$and": [{"film:genre:comedy": true}, {"film:year": {"$gte": 2000}}]}
   - Use $or for alternatives: {"$or": [{...}, {...}]}

4. IMPORTANT:
   - Only use filter names and values from the provided schema
   - Tag values must match EXACTLY (case-sensitive)
   - Return {} if no filter is appropriate
"""


class LLMQueryPlanner:
    """Two-step query planner for generating ChromaDB metadata filters."""

    def __init__(self, filter_catalog_path: Union[str, Path]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")

        lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
        dspy.configure(lm=lm)

        self.filter_catalog = self._load_catalog(filter_catalog_path)
        self.select_filters = dspy.Predict(SelectFiltersSignature)
        self.generate_where = dspy.Predict(GenerateWhereClauseSignature)

    def _load_catalog(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Filter catalog not found: {p}")
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load filter catalog: {e}")
            return {}

    def _format_available_filters(self) -> str:
        lines = []
        for name, info in sorted(self.filter_catalog.items()):
            lines.append(f"- {name} ({info.get('type', 'unknown')})")
        return "\n".join(lines)

    def _format_filter_schema(self, selected: List[str]) -> str:
        lines: List[str] = []
        for name in selected:
            info = self.filter_catalog.get(name)
            if not info:
                continue

            filter_type = info.get("type", "unknown")
            lines.append(f"\nFilter: {name}")
            lines.append(f"Type: {filter_type}")

            allowed = info.get("allowed_values", [])
            if filter_type == "bool" and allowed:
                # Show values with frequencies for bool tags
                values_str = []
                for v in allowed[:30]:
                    if isinstance(v, dict):
                        values_str.append(f"{v['value']} (freq: {v['frequency']})")
                    else:
                        values_str.append(str(v))
                if len(allowed) > 30:
                    values_str.append(f"... and {len(allowed) - 30} more")
                lines.append(f"Allowed values: {', '.join(values_str)}")
                lines.append(f"Usage: {{\"{name}:VALUE\": true}} for HAS, {{\"{name}:VALUE\": false}} for NOT")
            elif filter_type in ("int", "float") and allowed:
                numeric_vals = [v for v in allowed if isinstance(v, (int, float))]
                if numeric_vals:
                    lines.append(f"Range: {min(numeric_vals)} to {max(numeric_vals)}")
                    lines.append(f"Sample values: {numeric_vals[:10]}")
                lines.append(f"Usage: {{\"{name}\": {{\"$eq\": VALUE}}}} or {{\"{name}\": {{\"$gte\": VALUE}}}}")

        return "\n".join(lines)

    def _normalize_where(self, clause: dict) -> Optional[dict]:
        """
        Ensure where clause is valid for ChromaDB:
         - If empty, return None
         - If single condition, return it directly
         - If multiple conditions, wrap in $and
         - If already a valid logical op, pass through
        """
        if not clause:
            return None

        # If already a top-level logical operator with >=2 items, pass through
        for op in ("$and", "$or"):
            if op in clause:
                items = clause[op]
                if isinstance(items, list) and len(items) >= 2:
                    return clause
                elif isinstance(items, list) and len(items) == 1:
                    # Unwrap single-item logical op
                    return items[0]
                else:
                    return None

        # Flatten simple conditions into a list
        items = [{k: v} for k, v in clause.items()]

        if len(items) == 0:
            return None
        elif len(items) == 1:
            return items[0]
        else:
            return {"$and": items}

    def plan(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a valid ChromaDB where clause, or None if no filters needed."""
        if not self.filter_catalog:
            return None

        # Step 1: Select relevant filters
        available = self._format_available_filters()
        step1 = self.select_filters(query=query, available_filters=available)
        selected = _parse_json(step1.selected_filters, default=[])

        if not selected:
            logger.info(f"No filters selected for query: {query}")
            return None

        logger.info(f"Selected filters: {selected}")

        # Step 2: Generate where clause
        schema_str = self._format_filter_schema(selected)
        step2 = self.generate_where(
            query=query,
            filter_schema=schema_str,
            chroma_instructions=CHROMA_WHERE_INSTRUCTIONS,
        )

        raw_clause = _parse_json(step2.where_clause, default={})
        if not isinstance(raw_clause, dict):
            logger.warning("LLM returned invalid where clause format")
            return None

        clause = self._normalize_where(raw_clause)

        if clause:
            logger.info(f"Where clause: {json.dumps(clause)}")

        return clause
