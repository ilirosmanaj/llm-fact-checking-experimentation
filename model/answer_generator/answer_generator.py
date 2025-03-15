from utils.utils import *
from pipeline import *
from abc import abstractmethod
from typing import List

class AnswerGenerator(PipelineLLM):
    """
    A class for generating answers using a Large Language Model (LLM).

    Inherits:
        PipelineLLM: Provides the base LLM pipeline functionality

    Purpose:
        The AnswerGenerator class is an abstract class for All Answer Generators.

    Methods:
        - forward(data: str) -> str: An abstract method to be implemented in subclasses, defining
                                     how the input data is processed to generate an answer.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        super().__init__(config)

    @abstractmethod
    def forward(self, data: str) -> str:
        """
        Generate Answer from the input string
        """
        raise NotImplementedError

    @property
    def input_output_format(self):
        return {
            "input": ["reference_documents", "question"],
            "output": ["generated_answer"],
        }
