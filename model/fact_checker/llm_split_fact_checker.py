from model.fact_checker.fact_checker import *
from pipeline import *


class LLMSplitFactChecker(FactChecker, PipelineLLM, PipelinePrompt):
    class LLMSplitFactChecker:
        """
        LLMSplitFactChecker is designed to compare answer triplets with reference triplets using a language model.
        The model compares a answer triplet with the reference triplets at each request and returns the merged comparison result.


        Attributes:
            config (dict): Configuration dictionary for initializing the class.

        Methods:
            directions:
                Property that returns a list of directions for the output format.

            forward(answer_triplets, reference_triplets):
                Compares answer triplets with reference triplets and returns the comparison results.

            get_model_prompt(answer_triplets, reference_triplets, **kwargs):
                Generates a model prompt for comparing answer triplets with reference triplets.

            splitted_triplet_comparison_input_formatter(answer_triplets, reference_triplets):
                Formats the input for the triplet comparison.

            parse_splitted_triplet_comparison_output(string_output):
                Parses the output from the model and returns the comparison result.
        """

    def __init__(self, config: dict, logger: logging.Logger):
        FactChecker.__init__(self, config, logger)
        PipelineLLM.__init__(self, config)
        PipelinePrompt.__init__(self, config)

    @property
    def directions(self):
        return [
            "Answer only for the output",
            "output should be triplet_idx:result",
            "the output should be created along the input triplets",
            "the result should be one of True or False",
        ]

    def forward(self, answer_triplets: list, reference_triplets: list):
        """
        Compares all answer triplet with reference triplets using a model and returns the comparison results.
        In one request, the model compares one answer triplet with all reference triplets.

        Args:
            answer_triplets (list): A list of triplets representing the answers to be compared.
            reference_triplets (list): A list of triplets representing the reference data for comparison.

        Returns:
            tuple: A dictionary where keys are indices and values are parsed comparison results, and None.
        """
        comparison_result = {}

        for idx, answer_triplets in enumerate(answer_triplets):
            splitted_triplet_comparison_prompt = self.get_model_prompt(
                answer_triplets=answer_triplets,
                reference_triplets=reference_triplets,
            )
            match_result = self.model.invoke(splitted_triplet_comparison_prompt).content
            parsed_output = self.parse_splitted_triplet_comparison_output(
                match_result, answer_triplets
            )
            comparison_result[idx] = parsed_output
        return comparison_result, None

    def get_model_prompt(
        self, answer_triplets: list, reference_triplets: list, **kwargs
    ):
        """
        Generates a model prompt based on the provided answer and reference triplets.

        Args:
            answer_triplets (list): A list of triplets representing the answer.
            reference_triplets (list): A list of triplets representing the reference.
            **kwargs: Additional keyword arguments.

        Returns:
            message_list: The formatted model prompt.
        """

        return self.message_list_template["triplet_match_test"].invoke(
            input=self.splitted_triplet_comparison_input_formatter(
                answer_triplets=answer_triplets, reference_triplets=reference_triplets
            )
        )

    def splitted_triplet_comparison_input_formatter(
        self, answer_triplets: list, reference_triplets: list
    ):
        """
        Formats the input for comparing answer triplets with reference triplets.

        Args:
            answer_triplets (list): A list of triplets representing the answer.
            reference_triplets (list): A list of triplets representing the reference.

        Returns:
            dict: A dictionary containing formatted directions, answer triplets, and reference triplets.
        """
        return {
            "directions": "\n-".join(self.directions),
            "answer_triplets": f"0: {answer_triplets}",
            "reference_triplets": "\n-".join(
                [str(source_triplet) for source_triplet in reference_triplets]
            ),
        }

    def parse_splitted_triplet_comparison_output(self, string_output, answer_triplets):
        """
        Parses the output string from a triplet comparison and extracts the final answer.

        Args:
            string_output (str): The output string containing the triplet comparison results.
            answer_triplets (list): The answer triplets used for comparison.

        Returns:
            list or bool: The parsed final answer as a list if successful, otherwise False.
        """
        try:
            answer_part = string_output.split("[FINAL ANSWER]")[-1]
            splitted_string_outputs = answer_part.split(":")[-1].strip()

            return eval(splitted_string_outputs)
        except Exception as e:
            self.logger.warning(
                "Failed to parse the splitted fact checker output. : %s", str(e)
            )
            self.logger.debug("Error occured in : %s", string_output)
            self.logger.debug("Answer triplets: %s", answer_triplets)
            return False
