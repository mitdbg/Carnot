import logging

from carnot.core.elements.records import DataRecord
from carnot.core.models import PlanStats
from carnot.execution.execution_strategy import ExecutionStrategy
from carnot.operators.scan import ContextScanOp, ScanPhysicalOp
from carnot.optimizer.plan import PhysicalPlan
from carnot.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class SequentialSingleThreadExecutionStrategy(ExecutionStrategy):
    """
    A single-threaded execution strategy that processes operators sequentially.
    
    This strategy processes all records through one operator completely before moving to the next operator
    in the execution plan. For example, if we have operators A -> B -> C and records [1,2,3]:
    1. First processes records [1,2,3] through operator A
    2. Then takes A's output and processes all of it through operator B
    3. Finally processes all of B's output through operator C
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 1

    def _execute_plan(self, plan: PhysicalPlan, input_queues: dict[str, dict[str, list]], plan_stats: PlanStats) -> tuple[list[DataRecord], PlanStats]:
        # execute the plan one operator at a time
        output_records = []
        for topo_idx, operator in enumerate(plan):
            # if we've filtered out all records, terminate early
            source_unique_full_op_ids = (
                [f"source_{operator.get_full_op_id()}"]
                if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                else plan.get_source_unique_full_op_ids(topo_idx, operator)
            )
            unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
            num_inputs = sum(len(input_queues[unique_full_op_id][source_unique_full_op_id]) for source_unique_full_op_id in source_unique_full_op_ids)
            if num_inputs == 0:
                break

            # begin to process this operator
            records, record_op_stats = [], []
            logger.info(f"Processing operator {operator.op_name()} ({unique_full_op_id})")

            # process records according to operator type
            source_unique_full_op_id = source_unique_full_op_ids[0]
            for input_record in input_queues[unique_full_op_id][source_unique_full_op_id]:
                record_set = operator(input_record)
                records.extend(record_set.data_records)
                record_op_stats.extend(record_set.record_op_stats)
                num_outputs = sum(record._passed_operator for record in record_set.data_records)

                # update the progress manager
                self.progress_manager.incr(unique_full_op_id, num_inputs=1, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # update plan stats
            plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

            # update next input_queue (if it exists)
            output_records = [record for record in records if record._passed_operator]
            next_unique_full_op_id = plan.get_next_unique_full_op_id(topo_idx, operator)
            if next_unique_full_op_id is not None:
                input_queues[next_unique_full_op_id][unique_full_op_id] = output_records

            logger.info(f"Finished processing operator {operator.op_name()} ({unique_full_op_id}), and generated {len(records)} records")

        # finalize plan stats
        plan_stats.finish()

        return output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues for each operation
        input_queues = self._create_input_queues(plan)

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, num_samples=self.num_samples, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
