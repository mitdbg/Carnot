from enum import Enum

from carnot.execution.parallel_execution_strategy import ParallelExecutionStrategy
from carnot.execution.single_threaded_execution_strategy import SequentialSingleThreadExecutionStrategy


class ExecutionStrategyType(Enum):
    """Available execution strategy types"""
    SEQUENTIAL = SequentialSingleThreadExecutionStrategy
    PARALLEL = ParallelExecutionStrategy

    def is_fully_parallel(self) -> bool:
        """Check if the execution strategy executes operators in parallel."""
        return self == ExecutionStrategyType.PARALLEL
