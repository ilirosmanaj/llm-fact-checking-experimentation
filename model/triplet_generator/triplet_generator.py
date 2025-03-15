from utils.utils import *
from pipeline import *
from abc import abstractmethod
from typing import List, Tuple

class TripletGenerator(PipelineBase):
    """
    TripletGenerator is a base class for generating triplets from input data. It inherits from PipelineBase and requires a configuration dictionary during initialization.

    Methods:
        __init__(config: dict):
            Initializes the TripletGenerator with the given configuration.

        forward(data: str) -> List[Tuple[str, str, str]]:
            Abstract method that must be implemented by subclasses to generate triplets from the input data. Raises NotImplementedError if not overridden.

    Properties:
        input_output_format:
            Returns a dictionary specifying the expected input and output format for the triplet generation process.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config)
        self.logger = logger

    @abstractmethod
    def forward(self, data: str) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from the data
        """
        raise NotImplementedError

    @property
    def input_output_format(self):
        return {
            "input": ["generated_answer"],
            "output": ["answer_triplets"],
        }
