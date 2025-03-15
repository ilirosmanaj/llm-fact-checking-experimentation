from pipeline import *
from dataset import BaseDataset
from utils.utils import *


class DemonstrationDataset(BaseDataset, PipelineLLM, PipelineDemonstration):
    """
    DemonstrationDataset is a class that handles the generation, saving, and loading of demonstration data for a model.

    Methods:
        __init__(self, config, model):
            Initializes the DemonstrationDataset with the given configuration and model.

        get_demo_prompt(self, demo_input: dict, demo_output: list) -> str:

        generate_demo_data(self, idx: int, prompt_template: dict, save_data=True, **kwargs) -> str:
            Generates demonstration data based on the provided prompt template and saves it if specified.

        remove_instructions(self, text: str) -> str:
            Removes instructions from the given text.

        get_demo_dataset(self) -> dict:
            Loads and returns the demonstration dataset from the specified path.

        save_dataset_as_files(self, dataset: dict, path: str):
            Saves the given dataset to files at the specified path.

        get_perfect_scored_generation_samples(self, data_path: str) -> list:
            Retrieves samples with perfect scores from the prediction data file.

        extract_demo_data_features(self, data_dict: dict) -> dict:
            Extracts and returns the input and output features from the given data dictionary.

        get_manual_demo_samples(self, data_path: str) -> list:
            Loads and returns manually created demonstration samples from the specified path.

    All the methods related to generate, save, load demonstration data
    """

    def __init__(self, config: dict, model):
        BaseDataset.__init__(self, config)
        PipelineLLM.__init__(self, config)
        PipelineDemonstration.__init__(self, config)
        self.demo_dataset = self.get_demo_dataset()
        self.model = model

    def get_demo_prompt(self, demo_input: dict, demo_output: list):
        """
        Returns a demonstration prompt formatted as a few-shot example block.

        Args:
            examples (list of dict): A list of dictionaries, each containing
                'idx' (int): The example index,
                'demo_input' (dict): The input data for the demonstration example,
                'demo_output' (list): The expected outputs (e.g. golden answer) for the demonstration example.

        Returns:
            str: A text-based demonstration prompt in the specified format.
        """

        prompt = f"Input:\n{demo_input}\nOutput:\n{demo_output}\n\n"
        return prompt

    def generate_demo_data(
        self,
        idx: int,
        prompt_template: dict,
        save_data=True,
        **kwargs,
    ) -> str:
        """
        Generates demonstration data based on the provided prompt template and additional arguments.

        Args:
            idx (int): The index of the demonstration data.
            prompt_template (dict): A dictionary containing the prompt template.
            save_data (bool, optional): A flag indicating whether to save the generated data. Defaults to True.
            **kwargs: Additional keyword arguments containing 'input' and 'output' data.

        Returns:
            str: The generated demonstration example.
        """
        demo_data = prompt_template(**kwargs["input"])[1].content  # extract original
        demo_data = self.remove_instructions(demo_data)
        demo_input = kwargs["input"]
        demo_output = list(kwargs["output"].values())
        demo_example = self.get_demo_prompt(demo_data, str(demo_output[0]))
        if save_data:
            self.save_dataset_as_files(
                {
                    idx: {
                        "text": demo_example,
                        "input": demo_input,
                        "output": demo_output,
                    }
                },
                path=f"{self.config.path.data.base}{self.config.path.data.demo}",
            )
        return demo_example

    def remove_instructions(self, text):
        """
        Removes instructions from the given text.

        This method processes the input text by splitting it into lines and
        collecting lines until a line containing the word "Task" is encountered.
        Lines after the first occurrence of "Task" are ignored.

        Args:
            text (str): The input text from which instructions need to be removed.

        Returns:
            str: The text with instructions removed, joined by newline characters.
        """

        data_part = []

        for splitted_text in text.split("\n"):
            if "Task" not in splitted_text:
                data_part.append(splitted_text)
            else:
                break
        return "\n".join(data_part)

    def get_demo_dataset(self):
        """
        Retrieves the demonstration dataset from the specified path in the configuration.

        This method checks if the demonstration dataset path exists. If it does not exist,
        it creates the necessary directories. It then loads the dataset files for each
        demo type specified in the configuration and returns them as a dictionary.

        Returns:
            dict: A dictionary where the keys are the demo types and the values are the
                  loaded datasets. If the dataset path does not exist, it returns an
                  empty dictionary for each demo type.
        """
        demo_dataset_path = f"{self.config.path.data.base}{self.config.path.data.demo}"
        if not os.path.exists(demo_dataset_path):
            os.makedirs(demo_dataset_path)

        if os.path.exists(demo_dataset_path):
            demo = {}
            for demo_type in self.config.model:
                type_demo_dataset_path = f"{demo_dataset_path}/{demo_type}"
                if not os.path.exists(type_demo_dataset_path):
                    os.makedirs(type_demo_dataset_path)
                demo[demo_type] = self.load_files_as_dataset(type_demo_dataset_path)
                return demo
        else:
            return {demo_type: {} for demo_type in self.config.model}

    def save_dataset_as_files(self, dataset: dict, path: str):
        """
        Save the given dataset to files at the specified path.

        Args:
            dataset (Any): The dataset to be saved.
            path (str): The directory path where the dataset files will be saved.

        Returns:
            Any: The result of the superclass's save_dataset_as_files method.
        """
        demo_path = f"{path}/{self.config.demo_target_model}"
        return super().save_dataset_as_files(dataset, demo_path)

    def get_perfect_scored_generation_samples(self, data_path):
        """
        Extracts and returns samples with perfect scores from the prediction data.

        This method reads a JSON file containing prediction data, filters out the samples
        that have a precision score of 1.0 and have more than one fact check prediction,
        and then extracts the relevant features from these samples.

        Args:
            data_path (str): The path to the directory containing the prediction data file.

        Returns:
            list: A list of dictionaries, each containing the extracted features of a sample
                  that has a perfect precision score and more than one fact check prediction.
        """

        prediction_file_path = (
            f"{data_path}/{self.config.path.experiment_result.predictions}"
        )
        prediction_data_file = json.load(open(prediction_file_path))
        full_scored_samples = [
            self.extract_demo_data_features(prediction_dict)
            for prediction_dict in prediction_data_file
            if prediction_dict["precision"] == 1.0
            and len(prediction_dict["fact_check_prediction_binary"]) > 1
        ]
        return full_scored_samples

    def extract_demo_data_features(self, data_dict):
        """
        Extracts input and output features from the provided data dictionary based on the model's input-output format. the input-output format is defined in the abstract class of each model.

        Args:
            data_dict (dict): A dictionary containing the data from which features are to be extracted.

        Returns:
            dict: A dictionary with two keys, "input" and "output". Each key maps to another dictionary containing the respective features as specified in the model's input-output format.
        """

        return {
            "input": {
                feature: data_dict[feature]
                for feature in self.model.input_output_format["input"]
            },
            "output": {
                feature: data_dict[feature]
                for feature in self.model.input_output_format["output"]
            },
        }

    def get_manual_demo_samples(self, data_path):
        """
        Loads and processes manual demonstration samples from a JSON file.

        Args:
            data_path (str): The path to the JSON file containing the manual demonstration data.

        Returns:
            list: A list of processed demonstration data features extracted from the JSON file.
        """

        manual_demo_data_path = f"{data_path}"
        manual_demo_data_file = json.load(
            open(manual_demo_data_path, "r", encoding="utf-8")
        )
        return [
            self.extract_demo_data_features(demo_data_dict)
            for demo_data_dict in manual_demo_data_file
        ]
