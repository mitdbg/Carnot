from __future__ import annotations

import logging
from copy import deepcopy

from pydantic.fields import FieldInfo

from carnot.constants import Model
from carnot.core.data.dataset import Dataset
from carnot.core.lib.schemas import get_schema_field_names
from carnot.execution.execution_strategy_type import ExecutionStrategyType
from carnot.operators.logical import (
    ComputeOperator,
    SearchOperator,
)
from carnot.optimizer import (
    IMPLEMENTATION_RULES,
    TRANSFORMATION_RULES,
)
from carnot.optimizer.cost_model import BaseCostModel, SampleBasedCostModel
from carnot.optimizer.optimizer_strategy_type import OptimizationStrategyType
from carnot.optimizer.plan import PhysicalPlan
from carnot.optimizer.primitives import Group, LogicalExpression
from carnot.optimizer.tasks import (
    ApplyRule,
    ExploreGroup,
    OptimizeGroup,
    OptimizeLogicalExpression,
    OptimizePhysicalExpression,
)
from carnot.policy import Policy

logger = logging.getLogger(__name__)


class Optimizer:
    """
    The optimizer is responsible for searching the space of possible physical plans
    for a user's initial (logical) plan and selecting the one which is closest to
    optimizing the user's policy objective.

    This optimizer is modeled after the Cascades framework for top-down query optimization:
    - Thesis describing Cascades implementation (Chapters 1-3):
      https://15721.courses.cs.cmu.edu/spring2023/papers/17-optimizer2/xu-columbia-thesis1998.pdf

    - Andy Pavlo lecture with walkthrough example: https://www.youtube.com/watch?v=PXS49-tFLcI

    - Original Paper: https://www.cse.iitb.ac.in/infolab/Data/Courses/CS632/2015/Papers/Cascades-graefe.pdf

    Notably, this optimization framework has served as the backbone of Microsoft SQL Server, CockroachDB,
    and a few other important DBMS systems.

    NOTE: the optimizer currently assumes that field names are unique across schemas; we do try to enforce
          this by rewriting field names underneath-the-hood to be "{schema_name}.{field_name}", but this still
          does not solve a situation in which -- for example -- a user uses the pz.URL schema twice in the same
          program. In order to address that situation, we will need to augment our renaming scheme.
    """

    def __init__(
        self,
        policy: Policy,
        cost_model: BaseCostModel,
        available_models: list[Model],
        join_parallelism: int = 64,
        reasoning_effort: str | None = None,
        api_base: str | None = None,
        verbose: bool = False,
        optimizer_strategy: OptimizationStrategyType = OptimizationStrategyType.PARETO,
        execution_strategy: ExecutionStrategyType = ExecutionStrategyType.PARALLEL,
        use_final_op_quality: bool = False, # TODO: make this func(plan) -> final_quality
        **kwargs,
    ):
        # store the policy
        self.policy = policy

        # store the cost model
        self.cost_model = cost_model

        # mapping from each group id to its Group object
        self.groups = {}

        # mapping from each expression to its Expression object
        self.expressions = {}

        # the stack of tasks to perform during optimization
        self.tasks_stack = []

        # the lists of implementation and transformation rules that the optimizer can apply
        self.implementation_rules = IMPLEMENTATION_RULES
        self.transformation_rules = TRANSFORMATION_RULES

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

        # remove transformation rules for optimization strategies which do not require them
        if optimizer_strategy.no_transformation():
            self.transformation_rules = []

        # if we are not performing optimization, set available models to be single model
        # and remove all optimizations (except for bonded queries)
        if optimizer_strategy == OptimizationStrategyType.NONE:
            self.available_models = [available_models[0]]

        # store optimization hyperparameters
        self.verbose = verbose
        self.available_models = available_models
        self.join_parallelism = join_parallelism
        self.reasoning_effort = reasoning_effort
        self.api_base = api_base
        self.optimizer_strategy = optimizer_strategy
        self.execution_strategy = execution_strategy
        self.use_final_op_quality = use_final_op_quality

        logger.info(f"Initialized Optimizer with verbose={self.verbose}")
        logger.debug(f"Initialized Optimizer with params: {self.__dict__}")

    def update_cost_model(self, cost_model: BaseCostModel):
        self.cost_model = cost_model

    def get_physical_op_params(self):
        return {
            "verbose": self.verbose,
            "available_models": self.available_models,
            "join_parallelism": self.join_parallelism,
            "reasoning_effort": self.reasoning_effort,
            "api_base": self.api_base,
        }

    def deepcopy_clean(self):
        optimizer = Optimizer(
            policy=self.policy,
            cost_model=SampleBasedCostModel(),
            verbose=self.verbose,
            available_models=self.available_models,
            join_parallelism=self.join_parallelism,
            reasoning_effort=self.reasoning_effort,
            api_base=self.api_base,
            optimizer_strategy=self.optimizer_strategy,
            execution_strategy=self.execution_strategy,
            use_final_op_quality=self.use_final_op_quality,
        )
        return optimizer

    def update_strategy(self, optimizer_strategy: OptimizationStrategyType):
        # set the optimizer_strategy
        self.optimizer_strategy = optimizer_strategy

        # get the strategy class associated with the optimizer strategy
        optimizer_strategy_cls = optimizer_strategy.value
        self.strategy = optimizer_strategy_cls()

        # remove transformation rules for optimization strategies which do not require them
        if optimizer_strategy.no_transformation():
            self.transformation_rules = []

    def construct_group_tree(self, dataset: Dataset) -> tuple[int, dict[str, FieldInfo], dict[str, set[str]]]:
        logger.debug(f"Constructing group tree for dataset: {dataset}")
        ### convert node --> Group ###
        # create the op for the given node
        op = dataset._operator

        # compute the input group id(s) and field(s) for this node
        if len(dataset._sources) == 0:
            input_group_ids, input_group_fields, input_group_properties = ([], {}, {})
        elif len(dataset._sources) == 1:
            input_group_id, input_group_fields, input_group_properties = self.construct_group_tree(dataset._sources[0])
            input_group_ids = [input_group_id]
        elif len(dataset._sources) == 2:
            left_input_group_id, left_input_group_fields, left_input_group_properties = self.construct_group_tree(dataset._sources[0])
            right_input_group_id, right_input_group_fields, right_input_group_properties = self.construct_group_tree(dataset._sources[1])
            input_group_ids = [left_input_group_id, right_input_group_id]
            input_group_fields = {**left_input_group_fields, **right_input_group_fields}
            input_group_properties = deepcopy(left_input_group_properties)
            for k, v in right_input_group_properties.items():
                if k in input_group_properties:
                    input_group_properties[k].update(v)
                else:
                    input_group_properties[k] = deepcopy(v)
        else:
            raise NotImplementedError("Constructing group trees for datasets with more than 2 sources is not supported.")

        # compute the fields added by this operation and all fields
        input_group_short_field_names = list(
            map(lambda full_field: full_field.split(".")[-1], input_group_fields.keys())
        )
        new_fields = {
            field_name: op.output_schema.model_fields[field_name.split(".")[-1]]
            for field_name in get_schema_field_names(op.output_schema, id=dataset.id)
            if (field_name not in input_group_short_field_names) or (hasattr(op, "udf") and op.udf is not None)
        }
        all_fields = {**input_group_fields, **new_fields}

        # compute the set of (short) field names this operation depends on
        depends_on_field_names = (
            {} if dataset.is_root else {field_name.split(".")[-1] for field_name in op.depends_on}
        )

        # NOTE: group_id is computed as the unique (sorted) set of fields and properties;
        #       If an operation does not modify the fields (or modifies them in a way that
        #       can create an idential field set to an earlier group) then we must add an
        #       id from the operator to disambiguate the two groups.
        # compute all properties including this operations'
        all_properties = deepcopy(input_group_properties)
        if isinstance(op, ComputeOperator):
            op_instruction = op.instruction
            if "instructions" in all_properties:
                all_properties["instructions"].add(op_instruction)
            else:
                all_properties["instructions"] = set([op_instruction])

        elif isinstance(op, SearchOperator):
            op_search_query = op.search_query
            if "search_queries" in all_properties:
                all_properties["search_queries"].add(op_search_query)
            else:
                all_properties["search_queries"] = set([op_search_query])

        # construct the logical expression and group
        logical_expression = LogicalExpression(
            operator=op,
            input_group_ids=input_group_ids,
            input_fields=input_group_fields,
            depends_on_field_names=depends_on_field_names,
            generated_fields=new_fields,
            group_id=None,
        )
        group = Group(
            logical_expressions=[logical_expression],
            fields=all_fields,
            properties=all_properties,
        )
        logical_expression.set_group_id(group.group_id)

        # add the expression and group to the optimizer's expressions and groups and return
        self.expressions[logical_expression.expr_id] = logical_expression
        self.groups[group.group_id] = group
        logger.debug(f"Constructed group tree for dataset: {dataset}")
        logger.debug(f"Group: {group.group_id}, {all_fields}, {all_properties}")

        return group.group_id, all_fields, all_properties

    def convert_query_plan_to_group_tree(self, dataset: Dataset) -> str:
        logger.debug(f"Converting query plan to group tree for dataset: {dataset}")

        # compute depends_on field for every node
        short_to_full_field_name = {}
        for node in dataset:
            # update mapping from short to full field names
            short_field_names = get_schema_field_names(node.schema)
            full_field_names = get_schema_field_names(node.schema, id=node.id)
            for short_field_name, full_field_name in zip(short_field_names, full_field_names, strict=False):
                # set mapping automatically if this is a new field
                if short_field_name not in short_to_full_field_name or (hasattr(node._operator, "udf") and node._operator.udf is not None):
                    short_to_full_field_name[short_field_name] = full_field_name

            # if the node is a root Dataset, then skip
            if node.is_root:
                continue

            # If the node already has depends_on specified, then resolve each field name to a full (unique) field name
            if len(node._operator.depends_on) > 0:
                node._operator.depends_on = list(map(lambda field: short_to_full_field_name[field], node._operator.depends_on))
                continue

            # otherwise, make the node depend on all upstream nodes
            node._operator.depends_on = set()
            upstream_nodes = node.get_upstream_datasets()
            for upstream_node in upstream_nodes:
                upstream_field_names = get_schema_field_names(upstream_node.schema, id=upstream_node.id)
                node._operator.depends_on.update(upstream_field_names)
            node._operator.depends_on = list(node._operator.depends_on)

        # construct tree of groups
        final_group_id, _, _ = self.construct_group_tree(dataset)

        logger.debug(f"Converted query plan to group tree for dataset: {dataset}")
        logger.debug(f"Final group id: {final_group_id}")

        return final_group_id

    def heuristic_optimization(self, group_id: int) -> None:
        """
        Apply universally desirable transformations (e.g. filter/projection push-down).
        """
        pass

    def search_optimization_space(self, group_id: int) -> None:
        logger.debug(f"Searching optimization space for group_id: {group_id}")

        # begin the search for an optimal plan with a task to optimize the final group
        initial_task = OptimizeGroup(group_id)
        self.tasks_stack.append(initial_task)

        # TODO: conditionally stop when X number of tasks have been executed to limit exhaustive search
        while len(self.tasks_stack) > 0:
            task = self.tasks_stack.pop(-1)

            new_tasks = []
            if isinstance(task, (OptimizeGroup, ExploreGroup)):
                new_tasks = task.perform(self.groups)
            elif isinstance(task, OptimizeLogicalExpression):
                new_tasks = task.perform(self.transformation_rules, self.implementation_rules)
            elif isinstance(task, ApplyRule):
                context = {"costed_full_op_ids": self.cost_model.get_costed_full_op_ids()}
                new_tasks = task.perform(
                    self.groups, self.expressions, context=context, **self.get_physical_op_params(),
                )
            elif isinstance(task, OptimizePhysicalExpression):
                context = {"optimizer_strategy": self.optimizer_strategy, "execution_strategy": self.execution_strategy}
                new_tasks = task.perform(self.cost_model, self.groups, self.policy, context=context)

            self.tasks_stack.extend(new_tasks)

        logger.debug(f"Done searching optimization space for group_id: {group_id}")

    def optimize(self, dataset: Dataset) -> list[PhysicalPlan]:
        """
        The optimize function takes in an initial query plan and searches the space of
        logical and physical plans in order to cost and produce a (near) optimal physical plan.
        """
        logger.info(f"Optimizing query plan: {dataset}")
        # compute the initial group tree for the user plan
        dataset_copy = dataset.copy()
        final_group_id = self.convert_query_plan_to_group_tree(dataset_copy)

        # TODO
        # # do heuristic based pre-optimization
        # self.heuristic_optimization(final_group_id)

        # search the optimization space by applying logical and physical transformations to the initial group tree
        self.search_optimization_space(final_group_id)
        logger.info(f"Getting optimal plans for final group id: {final_group_id}")

        return self.strategy.get_optimal_plans(self.groups, final_group_id, self.policy, self.use_final_op_quality)
