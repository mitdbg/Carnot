import logging

from carnot.core.data.dataset import Dataset
from carnot.core.elements.records import DataRecord, DataRecordCollection
from carnot.core.models import ExecutionStats, PlanStats
from carnot.execution.execution_strategy import ExecutionStrategy
from carnot.optimizer.optimizer import Optimizer
from carnot.policy import Policy
from carnot.utils.hash_helpers import hash_for_id

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes queries through the complete pipeline:
    1. Optimization phase: Plan generation and selection
    2. Execution phase: Plan execution and result collection
    3. Result phase: Statistics gathering and result formatting
    """
    def __init__(
        self,
        dataset: Dataset,
        optimizer: Optimizer,
        execution_strategy: ExecutionStrategy,
        num_samples: int | None = None,
        scan_start_idx: int = 0,
        verbose: bool = False,
        progress: bool = True,
        max_workers: int | None = None,
        policy: Policy | None = None,
        available_models: list[str] | None = None,
        **kwargs,  # needed in order to provide compatibility with QueryProcessorConfig
    ):
        """
        Initialize QueryProcessor with optional custom components.
        
        Args:
            dataset: Dataset to process
            TODO
        """
        self.dataset = dataset
        self.optimizer = optimizer
        self.execution_strategy = execution_strategy
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.verbose = verbose
        self.progress = progress
        self.max_workers = max_workers
        self.policy = policy
        self.available_models = available_models

        if self.verbose:
            print("Available models: ", self.available_models)

        logger.info(f"Initialized QueryProcessor {self.__class__.__name__}")
        logger.debug(f"QueryProcessor initialized with config: {self.__dict__}")

    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"

        return hash_for_id(id_str)

    def _execute_best_plan(self, dataset: Dataset, optimizer: Optimizer) -> tuple[list[DataRecord], list[PlanStats]]:
        # get the optimal plan according to the optimizer
        plans = optimizer.optimize(dataset)
        final_plan = plans[0]

        # execute the plan
        records, plan_stats = self.execution_strategy.execute_plan(plan=final_plan)

        # return the output records and plan stats
        return records, [plan_stats]

    def execute(self) -> DataRecordCollection:
        logger.info(f"Executing {self.__class__.__name__}")

        # create execution stats
        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, self.optimizer)

        # update the execution stats to account for the work to execute the final plan
        execution_stats.add_plan_stats(plan_stats)
        execution_stats.finish()

        # construct and return the DataRecordCollection
        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info(f"Done executing {self.__class__.__name__}")

        return result
