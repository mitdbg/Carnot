import time

from carnot.core.models import OperatorStats
from carnot.data.dataset import Dataset


class LimitOperator:
    """Limit operator — truncates a dataset to the first *n* items.

    This is a purely deterministic operator with no LLM involvement.

    Representation invariant:
        - ``n >= 0``.

    Abstraction function:
        An instance of this class is a callable that, given a dataset, returns a new dataset
        containing at most ``n`` items (the first *n* in order).
    """
    def __init__(self, n: int, output_dataset_id: str):
        self.n = n
        self.output_dataset_id = output_dataset_id

    def __call__(self, dataset_id: str, input_datasets: dict[str, Dataset]) -> tuple[dict[str, Dataset], OperatorStats]:
        """Truncate the input dataset to the first *n* items.

        Requires:
            - *dataset_id* is a key in *input_datasets*.

        Returns:
            A tuple ``(output_datasets, stats)`` where *output_datasets*
            is a new ``dict[str, Dataset]`` with an additional entry
            keyed by ``self.output_dataset_id`` containing at most
            ``self.n`` items, and *stats* is an :class:`OperatorStats`
            with an empty ``llm_calls`` list (no LLM involvement).

        Raises:
            KeyError: If *dataset_id* is not in *input_datasets*.
        """
        op_start = time.perf_counter()

        # retrieve the input dataset
        input_dataset = input_datasets[dataset_id]
        
        # apply the limit operation to the dataset items
        results = input_dataset.items[:self.n]

        # create new dataset and return it with the input datasets
        output_dataset = Dataset(name=self.output_dataset_id, annotation=f"Limit operator output for n: {self.n}", items=results)
        output_datasets = {**input_datasets, output_dataset.name: output_dataset}

        op_stats = OperatorStats(
            operator_name="Limit",
            operator_id=self.output_dataset_id,
            wall_clock_secs=time.perf_counter() - op_start,
            llm_calls=[],
            items_in=len(input_dataset.items),
            items_out=len(results),
        )

        return output_datasets, op_stats
