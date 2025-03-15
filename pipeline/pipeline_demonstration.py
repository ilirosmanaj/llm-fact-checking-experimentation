import json

from pipeline.pipeline_prompt import PipelinePrompt
import random
from utils.utils import *
from glob import glob


class PipelineDemonstration(PipelinePrompt):
    """
    PipelineDemonstration is a class that extends PipelinePrompt and is used for generating and loading demonstration data for inference.

    Attributes:
        config (dict): Configuration dictionary.

    Methods:
        default_demo_output:
            Returns the default output for demonstration data.

        get_demo_data_by_idx(idx, num_samples, demo_type) -> str:
            Retrieves demonstration data by index, number of samples, and type of demonstration.

        random_demo_sampler(idx, demo_file_path, num_samples) -> list:

        format_demo_data(demo_list: list, demo_type: str) -> str:
            Formats the demonstration data based on the type of demonstration.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    @property
    def default_demo_output(self):
        return ""

    def get_demo_data_by_idx(
        self, idx: int, num_samples: int, demo_type: str, get_sample_idx=False
    ) -> str:
        """
        Retrieves and formats demonstration data based on the given index, number of samples, and demonstration type.

        Args:
            idx (int): The index to retrieve the demonstration data from.
            num_samples (int): The number of samples to retrieve.
            demo_type (str): The type of demonstration data to retrieve.

        Returns:
            str: The formatted demonstration data as a string.
        """

        demo_list = None
        demo_dataset_path = f"{self.config.path.data.base}{self.config.path.data.demo}"
        if os.path.exists(demo_dataset_path):
            type_demo_dataset_path = f"{demo_dataset_path}/{demo_type}"
            if os.path.exists(type_demo_dataset_path):

                demo_list, sampled_files = self.load_all_demos(
                    idx=idx,
                    demo_file_path=type_demo_dataset_path,
                )
        result = self.format_demo_data(demo_list, demo_type=demo_type)
        if get_sample_idx:
            return result, sampled_files
        else:
            return result

    def random_demo_sampler(self, idx, demo_file_path, num_samples) -> list:
        """
        Randomly samples JSON files from the provided list, loads their content, and returns the data.
        If num_samples exceeds the number of files in the list, fills the extra slots with empty dictionaries.

        Args:
            demo_file_path (str): Directory path where demo file exists.
            num_samples (int): Number of samples to return.

        Returns:
            list: A list containing the loaded JSON content and empty dictionaries if necessary.
        """
        demo_file_list = [
            i
            for i in glob(f"{demo_file_path}/*")
            if i.split("/")[-1].replace(".json", "") != idx
        ]
        # Shuffle and sample files (with replacement if num_samples exceeds list size)
        sampled_files = (
            random.choices(demo_file_list, k=num_samples)
            if num_samples > len(demo_file_list)
            else random.sample(demo_file_list, k=num_samples)
        )

        # Load the JSON content from sampled files
        result = []
        for file in sampled_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    result.append(json.load(f))
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                result.append(
                    self.default_demo_output
                )  # Append empty dict for files that fail to load
        return result, sampled_files

    def load_all_demos(self, idx, demo_file_path) -> list:
        """
        Loads all JSON files from the specified directory, excluding the one that matches 'idx',
        and returns them in the same output format as 'random_demo_sampler', but without random sampling.

        Args:
            idx (str): The identifier of the file to exclude.
            demo_file_path (str): Directory path where demo files exist.

        Returns:
            list: A list of loaded JSON content and the list of file paths.
        """
        demo_file_list = [
            i
            for i in glob(f"{demo_file_path}/*")
            if i.split("/")[-1].replace(".json", "") != idx
        ]

        # Instead of sampling, just use all files
        sampled_files = demo_file_list

        result = []
        for file in sampled_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    result.append(json.load(f))
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                result.append(self.default_demo_output)

        return result, sampled_files

    def format_demo_data(self, demo_list: list, demo_type: str) -> str:
        """
        Formats demonstration data based on the provided demo type.

        Args:
            demo_list (list): A list of demonstration data.
            demo_type (str): The type of demonstration data. Can be either
                             "hallucinated_data_generation" or "triplet_generator".

        Returns:
            str: A formatted string containing the demonstration data. If the
                 demo_list is None or empty after filtering, returns an empty string.
        """

        if demo_list is None:
            return ""
        else:
            if demo_type == "hallucinated_data_generation":
                demo_list = [i for i in demo_list if i != self.default_demo_output]
            else:
                demo_list = [
                    i["text"]
                    for i in demo_list
                    if i["text"] != self.default_demo_output
                ]

            if len(demo_list) == 0:
                return ""
            else:
                demo_text = "[BEGIN FEW-SHOT-EXAMPLES]\n"
                for idx, demo in enumerate(demo_list):
                    demo_text += f"<Example {idx} Input/Output Pair>\n{str(demo)}\n\n"

                demo_text += "[END FEW-SHOT-EXAMPLES]"
            return demo_text
