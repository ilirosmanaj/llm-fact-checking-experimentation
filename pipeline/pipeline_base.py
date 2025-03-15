class PipelineBase:
    """
    A base class for building data processing pipelines. Anything common to all pipelines should added here.

    Attributes:
        config (dict): A configuration dictionary containing parameters
                       for the pipeline's setup and behavior.
    """

    def __init__(self, config: dict):
        self.config = config

    @property
    def data_key_mapping(self):
        return {
            "question": "question",
            "generated_answer": "generated_answer",
            "answer_triplets": "answer_triplets",
            "reference_documents": "reference_documents",
            "reference_triplets": "reference_triplets",
            "fact_check_prediction_binary": "fact_check_prediction_binary",
            "question": "question",
        }
