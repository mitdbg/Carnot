# import time
# from collections.abc import Generator
# from dataclasses import dataclass
# from importlib import resources


# import yaml

from carnot.agents.tools import Tool

# from carnot.data.dataset import Dataset
from carnot.operators.code import CodeOperator  # , FinalAnswerStep


class ReasoningOperator(CodeOperator):
    """
    Represents a reasoning operator. For our purposes, this is a code operator with a prompt specialized
    for reasoning over the input datasets to produce an output dataset.
    """
    def __init__(self, task: str, model_id: str, tools: list[Tool] | None = None, additional_authorized_imports: list[str] | None = None, max_steps: int = 20):
        super().__init__(task, model_id, tools, additional_authorized_imports, max_steps)
        # self.prompt_templates = yaml.safe_load(
        #     resources.files("carnot.agents.prompts").joinpath("reasoning_operator.yaml").read_text()
        # )

