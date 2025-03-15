from model.answer_generator.base_llm_answer_generator import *
from model.answer_generator.llm_multishot_answer_generator import *

from model.triplet_generator.llm_multishot_triplet_generator import *
from model.triplet_generator.llm_triplet_generator import *

from model.fact_checker.llm_multishot_split_fact_checker import *
from model.fact_checker.llm_multishot_fact_checker import *
from model.fact_checker.partial_match_fact_checker import *
from model.fact_checker.exact_match_fact_checker import *
from model.fact_checker.llm_split_fact_checker import *
from model.fact_checker.llm_fact_checker import *

from model.hallucination_data_generator.llm_multishot_hallucination_data_generator import *
from model.hallucination_data_generator.llm_hallucination_data_generator import *

from model.reprompter.reprompter import *

model_name_class_mapping = {
    "answer_generator": {
        "base_llm": BaseLLMAnswerGenerator,
        "llm_n_shot": LLMMultiShotAnswerGenerator,
    },
    "triplet_generator": {
        "llm": LLMTripletGenerator,
        "llm_n_shot": LLMMultiShotTripletGenerator,
    },
    "fact_checker": {
        "exact_match": ExactMatchFactChecker,
        "partial_match": PartialMatchFactChecker,
        "llm": LLMFactChecker,
        "llm_split": LLMSplitFactChecker,
        "llm_n_shot": LLMMultiShotFactChecker,
        "llm_n_shot_split": LLMMultiShotSplitFactChecker,
    },
    "hallucination_data_generator": {
        "llm": LLMHallucinationDataGenerator,
        "llm_n_shot": LLMMultiShotHallucinationDataGenerator,
    },
    "reprompter": {"llm": Reprompter},
}
__all__ = [
    "BaseLLMAnswerGenerator",
    "LLMTripletGenerator",
    "ExactMatchFactChecker",
    "PartialMatchFactChecker",
    "LLMFactChecker",
    "HallucinationDataGenerator",
    "Reprompter",
    "LLMMultiShotAnswerGenerator",
    "LLMMultiShotFactChecker",
    "model_name_class_mapping",
]
