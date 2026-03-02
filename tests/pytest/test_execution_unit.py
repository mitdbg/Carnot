"""Unit tests for Execution helper methods (no LLM required).

Tests cover:
1. ``_get_op_from_plan_dict`` — returns the correct physical operator
   type for every known operator name, and raises for unknowns.
2. ``_get_ops_in_topological_order`` — produces the correct linearised
   order for single-op, chained, and branching (join) plan DAGs.
"""

from __future__ import annotations

import pytest

from carnot.data.dataset import Dataset
from carnot.execution.execution import Execution
from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator

# ── Helpers ─────────────────────────────────────────────────────────────────

# Minimal llm_config so Execution can be instantiated without env vars.
_LLM_CONFIG = {"OPENAI_API_KEY": "test-key-not-real"}


def _make_execution(datasets: list[Dataset] | None = None) -> Execution:
    """Create a minimal Execution instance for unit-testing helper methods.

    Requires:
        - ``_LLM_CONFIG`` provides a dummy API key.

    Returns:
        An ``Execution`` object ready for plan-parsing tests.
    """
    return Execution(
        query="test query",
        datasets=datasets or [],
        llm_config=_LLM_CONFIG,
    )


def _leaf_plan(name: str) -> dict:
    """Build a leaf plan node representing a raw Dataset reference.

    The plan node has no operator param so ``_get_op_from_plan_dict``
    falls through to the dataset-name lookup.
    """
    return {
        "name": name,
        "output_dataset_id": name,
        "params": {},
        "parents": [],
    }


def _op_plan(
    operator: str,
    output_id: str,
    parents: list[dict],
    **extra_params,
) -> dict:
    """Build an operator plan node with given parents."""
    params = {"operator": operator, **extra_params}
    return {
        "name": output_id,
        "output_dataset_id": output_id,
        "params": params,
        "parents": parents,
    }


# ═══════════════════════════════════════════════════════════════════════
# _get_op_from_plan_dict
# ═══════════════════════════════════════════════════════════════════════


class TestGetOpFromPlanDict:
    """Verify that each operator name maps to the correct physical class."""

    def test_code_operator(self):
        """'Code' → CodeOperator."""
        ex = _make_execution()
        plan = _op_plan("Code", "code1", [], task="compute stats")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, CodeOperator)
        assert parent_ids == []

    def test_limit_operator(self):
        """'Limit' → LimitOperator."""
        ex = _make_execution()
        plan = _op_plan("Limit", "limit1", [], n=5)
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, LimitOperator)

    def test_semantic_filter(self):
        """'SemanticFilter' → SemFilterOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemFilterOperator)
        assert parent_ids == ["Movies"]

    def test_semantic_map(self):
        """'SemanticMap' → SemMapOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticMap", "map1", [leaf],
            field="sentiment", type="str", field_desc="overall sentiment",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemMapOperator)

    def test_semantic_flat_map(self):
        """'SemanticFlatMap' → SemFlatMapOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticFlatMap", "flatmap1", [leaf],
            field="keyword", type="str", field_desc="extracted keyword",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemFlatMapOperator)

    def test_semantic_agg(self):
        """'SemanticAgg' → SemAggOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticAgg", "agg1", [leaf],
            task="summarize reviews",
            agg_fields=[{"name": "summary", "type": "str"}],
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemAggOperator)

    def test_semantic_groupby(self):
        """'SemanticGroupBy' → SemGroupByOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticGroupBy", "gby1", [leaf],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemGroupByOperator)

    def test_semantic_join(self):
        """'SemanticJoin' → SemJoinOperator with two parent ids."""
        ex = _make_execution()
        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan = _op_plan("SemanticJoin", "join1", [left, right], condition="same movie")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemJoinOperator)
        assert parent_ids == ["Movies", "Reviews"]

    def test_semantic_topk(self):
        """'SemanticTopK' → SemTopKOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticTopK", "topk1", [leaf],
            search_str="best action", k=5, index_name="flat",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemTopKOperator)

    def test_dataset_reference(self):
        """A plan node whose name matches a dataset returns that Dataset."""
        ds = Dataset(name="Movies", annotation="film data")
        ex = _make_execution(datasets=[ds])
        plan = _leaf_plan("Movies")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert op is ds
        assert parent_ids == []

    def test_unknown_operator_raises(self):
        """An unrecognized operator name raises ValueError."""
        ex = _make_execution()
        plan = _op_plan("UnknownOp", "unk1", [])
        with pytest.raises(ValueError, match="Unknown operator"):
            ex._get_op_from_plan_dict(plan)


# ═══════════════════════════════════════════════════════════════════════
# _get_ops_in_topological_order
# ═══════════════════════════════════════════════════════════════════════


class TestGetOpsInTopologicalOrder:
    """Verify topological linearization of plan DAGs."""

    def test_single_leaf(self):
        """A single dataset node yields one entry."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        plan = _leaf_plan("Movies")
        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 1
        assert ops[0][0] is ds

    def test_linear_chain(self):
        """Dataset → Filter yields [Dataset, Filter] in order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 2
        # First is the dataset, second is the filter
        assert ops[0][0] is ds
        assert isinstance(ops[1][0], SemFilterOperator)

    def test_two_step_chain(self):
        """Dataset → Filter → Map yields three entries in correct order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        filt = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        mapped = _op_plan(
            "SemanticMap", "map1", [filt],
            field="label", type="str", field_desc="genre label",
        )
        ops = ex._get_ops_in_topological_order(mapped)
        assert len(ops) == 3
        assert ops[0][0] is ds
        assert isinstance(ops[1][0], SemFilterOperator)
        assert isinstance(ops[2][0], SemMapOperator)

    def test_join_has_both_parents_before_join(self):
        """A join node is preceded by both parent datasets."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")
        ex = _make_execution(datasets=[ds_a, ds_b])

        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan = _op_plan("SemanticJoin", "join1", [left, right], condition="same movie")

        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 3

        # The join must be last
        assert isinstance(ops[2][0], SemJoinOperator)

        # Both datasets appear before the join
        pre_join_types = {type(o[0]) for o in ops[:2]}
        assert Dataset in pre_join_types

    def test_parent_ids_propagated(self):
        """Parent dataset IDs are correctly threaded through the order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="x")
        ops = ex._get_ops_in_topological_order(plan)
        # The filter's parent_ids should be ["Movies"]
        _, parent_ids = ops[1]
        assert parent_ids == ["Movies"]
