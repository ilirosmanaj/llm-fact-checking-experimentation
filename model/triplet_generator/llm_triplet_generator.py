from model.triplet_generator.triplet_generator import *
from utils.utils import *
from pipeline import *
from typing import Optional

class LLMTripletGenerator(TripletGenerator, PipelineLLM, PipelinePrompt):
    """
    LLMTripletGenerator is a class that generates triplets from input data using a language model.

    Methods
    -------
    __init__(self, config: dict)
        Initializes the LLMTripletGenerator with the given configuration.

    forward(self, data: str) -> List[Tuple[str, str, str]]
        Generates triplets from the input data.

    default_triplet(self)
        Returns the default triplet.

    get_model_prompt(self, text_input=None, **kwargs)
        Creates a prompt for triplet generation using the provided text input or generated answer.

    triplet_generation_input_formatter(self, text_input)
        Formats the input text for triplet generation.

    parse_triplet_generation_output(self, triplet_generation_output)
        Parses the output text to extract triplets.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        TripletGenerator.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    def forward(
        self, data: str, return_prompt: bool = False
    ) -> List[Tuple[str, str, str]]:
        """
        Processes the input data to generate triplets using a model.

        Args:
            data (str): The input text data from which triplets are to be generated.

        Returns:
            List[Tuple[str, str, str]]: A list of triplets generated from the input data.
        """
        triplet_generation_prompt = self.get_model_prompt(text_input=data)
        triplet_generation_output = self.model.invoke(triplet_generation_prompt).content
        if return_prompt:
            return (
                self.parse_triplet_generation_output(triplet_generation_output),
                triplet_generation_prompt,
            )
        else:
            return self.parse_triplet_generation_output(triplet_generation_output)

    @property
    def default_triplet(self):
        return ["", "", ""]

    def get_model_prompt(self, text_input: Optional[str] = None, **kwargs):
        """
        Create a prompt for triplet generation using the provided text input.
        Args:
            text_input (Optional[str]): The input text for generating the prompt. Defaults to None.
            **kwargs: Additional keyword arguments. Must include 'generated_answer' if text_input is not provided.

        Returns:
            str: The generated prompt for triplet generation.

        Raises:
            AssertionError: If neither text_input nor 'generated_answer' in kwargs is provided.
        """
        if text_input == None:
            assert (
                "generated_answer" in kwargs
            ), "one of text_input input or generated_answer should be provided"
            text_input = kwargs["generated_answer"]
        return self.message_list_template["triplet_generation_test"].invoke(
            input=self.triplet_generation_input_formatter(text_input)
        )

    def triplet_generation_input_formatter(self, text_input: str) -> dict:
        """
        Formats the input text for triplet generation.

        Args:
            text_input (str): The input text to be formatted.

        Returns:
            dict: A dictionary containing the formatted input text with the key 'input_text'.
        """
        return {
            "input_text": text_input,
        }

    def parse_triplet_generation_output(self, triplet_generation_output: str) -> List:
        """
        Parse output text to triplets.
        Args:
            triplet_generation_output (str): The output text containing triplets separated by newlines.

        Returns:
            list: A list of triplets parsed from the output text. If parsing fails or any triplet does not contain exactly three elements, returns the default triplet.

        """
        try:
            result = [
                eval(str(triplet).replace("[", "").replace("]", ""))
                for triplet in triplet_generation_output.split("\n")
            ]
            for triplet in result:
                if len(triplet) != 3:
                    self.logger.warning(
                        "Some triplet failed in generation : %s", str(triplet)
                    )
            result = [
                triplet if len(triplet) == 3 else self.default_triplet
                for triplet in result
            ]
        except Exception as e:  # todo: how to handle this exception?
            self.logger.warning("Error parsing triplet generation output. : %s", str(e))
            try:
                result = eval(self.preprocess_output(triplet_generation_output))
                result = self.default_triplet * len(result)
            except Exception as e:

                result = self.default_triplet * len(1)
            self.logger.debug("Error occured in : %s", triplet_generation_output)
            self.logger.debug("So we used : %s", str(result))
        return result
