from model.answer_generator.answer_generator import *
from pipeline import *


class LLMMultiShotAnswerGenerator(AnswerGenerator, PipelineDemonstration):

    def __init__(self, config: dict, logger: logging.Logger):
        AnswerGenerator.__init__(self, config, logger)
        PipelineDemonstration.__init__(self, config)

    @property
    def directions(self):
        # optional
        return [
            "Please answer the following question",
            "Answer the question with the information provided in the reference documents",
            "If reference documents do not contains the evidence for answer, just answer there is no evidence",
        ]

    def forward(self, message_list: list) -> str:
        """
        Args:
            message_list (list): A list of messages to be processed by the model.

        Returns:
            str: The content generated by the model.
        """
        return self.model.invoke(message_list).content

    def get_model_prompt(self, reference_documents: List[str], question: str, **kwargs):
        examples = self.get_demo_data_by_idx(
            idx=9999,  # need to change here
            num_samples=self.config.model.answer_generator.num_shot,
            demo_type="answer_generator",
        )
        return self.message_list_template["n_shot_answer_generation"].invoke(
            input=self.question_input_formatter(
                reference_documents=reference_documents,
                question=question,
                examples=examples,
            )
        )

    def question_input_formatter(
        self, reference_documents: List[str], question: str, examples: str
    ):
        result = {
            "directions": "\n".join(self.directions),
            "reference_documents": "\n- ".join(reference_documents),
            "question": question,
            "examples": examples,
        }
        print(result)
        return result
