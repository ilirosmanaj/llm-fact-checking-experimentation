from pipeline import *
from utils.utils import *


class HallucinationDataGenerator(PipelineLLM, PipelinePrompt):
    """
    HallucinationDataGenerator is used to generate data for hallucination detection in language models.

    Attributes:
        config (dict): Configuration dictionary for initializing the class.

    Methods:
        __init__(config: dict):
            Initializes the HallucinationDataGenerator with the given configuration.

        input_output_format:
            Returns the expected input and output format for the data generator.
    Note:
        hallucination dataset looks like this:
    {
        "question_id" : {
            "generated_non_hlcntn_answer": "generated_non_hlcntn_answer",
            "generated_answer": "hallucination answer",
            "non_hlcntn_triplets":[triplet_1, triplet_2, ...],
            "answer_triplets":[triplet_1, triplet_2, ...], \\the hallucinated answer triplets
            "hlcntn_triplet_index": [boolean list of len(answer_triplets) where True indicates the triplet is hallucinated],
            "hlcntn_part: str, \\string description of the hallucinated part of the answer
            "reference_documents": list of reference documents
            }
    }
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    @property
    def input_output_format(self):
        return {
            "input": ["reference_documents", "question"],
            "output": ["generated_answer"],
        }
