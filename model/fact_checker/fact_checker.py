from utils.utils import *
from pipeline import *
from abc import abstractmethod
from typing import List, Tuple

class FactChecker(PipelineBase):
    """
    FactChecker is a base(abstract) class for implementing fact-checking pipelines. It inherits from PipelineBase and requires a configuration dictionary during initialization.

    Methods:

    __init__(config: dict)
        Initializes the FactChecker with the given configuration.

    forward(input_triplet_list: List[List], label_triplets: List[List]) -> Tuple[dict, dict]
        Abstract method to generate triplets from the data. Must be implemented by subclasses.

    check_triplet_exists_in_dataset(triplet: List[List], source_triplets: List[List])
        Abstract method to check if a triplet exists in the dataset. Must be implemented by subclasses.

    Properties:
    input_output_format
        Returns the expected input and output format for the fact-checking pipeline.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config)
        self.logger = logger

    @abstractmethod
    def forward(
        self,
        answer_triplets: List[List],
        reference_triplets: List[List],
    ) -> Tuple[dict, dict]:
        """
        Generate triplets from the data
        """
        raise NotImplementedError

    @abstractmethod
    def check_triplet_exists_in_dataset(
        self, triplet: List[List], source_triplets: List[List]
    ):
        """
        method which checks if a triplet exists in the dataset
        """
        raise NotImplementedError

    @property
    def input_output_format(self):
        return {
            "input": ["answer_triplets", "reference_triplets"],
            "output": ["fact_check_prediction_binary"],
        }

    def flatten_triplets(self, triplet_segments: List[List[str]]) -> List[str]:
        """
        Flatten the list of triplets into a single list of strings.
        """
        return [triplet for sublist in triplet_segments for triplet in sublist]

    def merge_segment_outputs(self, output_list):
        if not output_list:
            self.logger.error("Empty fect check output list")

        # check if all dictionaries have the same keys
        keys_list = [set(d.keys()) for d in output_list]
        if not all(keys == keys_list[0] for keys in keys_list):
            self.logger.warning("Not all dictionaries have the same keys.")
            self.logger.debug("Keys: %s", keys_list)
            return {}

        all_keys = set().union(*keys_list)
        merged_fact_check_result = {key: False for key in all_keys}

        # merge the dictionaries
        for d in output_list:
            for key, value in d.items():
                if (
                    value
                ):  # if any of the dictionaries has the key as True, set the result to True
                    merged_fact_check_result[key] = True

        return merged_fact_check_result
