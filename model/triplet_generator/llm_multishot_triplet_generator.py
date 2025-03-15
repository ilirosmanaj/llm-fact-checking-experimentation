from model.triplet_generator.triplet_generator import *
from utils.utils import *
from pipeline import *


class LLMMultiShotTripletGenerator(
    TripletGenerator, PipelineLLM, PipelineDemonstration
):
    """
    LLMMultiShotTripletGenerator is a class that generates triplets from input data using a language model with multishot scheme.

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
        PipelineDemonstration.__init__(self, config)

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

    def get_model_prompt(self, text_input: str):
        """
        Create a prompt for triplet generation using the provided text input.

        This method constructs a prompt for generating triplets by invoking a
        template from the message list. It includes the input text and a set of
        example data for demonstration purposes.

        Args:
            text_input (str): The input text to be used for generating the prompt.

        Returns:
            str: The generated prompt for triplet generation.
        """
        examples = self.get_demo_data_by_idx(
            idx=9999,  # need to change here
            num_samples=self.config.model.triplet_generator.num_shot,
            demo_type="triplet_generator",
        )

        return self.message_list_template["n_shot_triplet_generation"].invoke(
            input={
                "input_text": text_input,
                "examples": examples,
            }
        )

    def parse_triplet_generation_output(self, triplet_generation_output: str) -> List:
        """
        Parse output text to triplets.

        Args:
            triplet_generation_output (str): The raw output text from the triplet generation process.

        Returns:
            list: A list of triplets parsed from the output text. Each triplet is a list of three elements.
                  If the output cannot be parsed correctly, a list of default triplets is returned.
        """
        try:
            result = eval(self.preprocess_output(triplet_generation_output))
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

                result = self.default_triplet * 1
            self.logger.debug("Error occured in : %s", triplet_generation_output)
            self.logger.debug("So we used : %s", str(result))
        return result

    def preprocess_output(self, output: str) -> str:
        """
        Preprocesses the given output string by performing a series of string replacements.

        Args:
            output (str): The output string to preprocess.

        Returns:
            str: The preprocessed output string with specific characters and patterns replaced.
        """
        example = (
            output.strip()
            .replace("{", "")
            .replace("}", "")
            .replace("]]]", "]]")
            .replace("]].", "]]")
        )
        return example
