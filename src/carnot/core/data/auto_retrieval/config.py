from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class Config:
    """Top-level configuration for SearchClient and internal modules."""
    chroma_uri: str
    collection_name: str
    space_budget_mb: int
    max_latency_ms: int
    # add more fields as needed


def load_config(path: str) -> Config:
    """Load a Config object from a YAML/JSON file."""
    # TODO: parse YAML/JSON here
    pass
