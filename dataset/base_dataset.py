from utils.utils import *
from pipeline import *
import json
from glob import glob

class BaseDataset(PipelineBase):
    """
    BaseDataset is a class that provides functionality to load and save datasets from and to JSON files.

    Methods:
        __init__(config: dict):
            Initializes the BaseDataset instance with the given configuration.
        load_files_as_dataset(path: str) -> dict:
            Loads JSON files from a specified directory and returns them as a dataset.
        save_dataset_as_files(dataset: dict, path: str) -> None:
            Saves the given dataset as individual JSON files in the specified directory.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def load_files_as_dataset(self, path: str):
        """
        Load JSON files from a specified directory and return them as a dataset.
        Args:
            path (str): The directory path containing JSON files to be loaded.
        Returns:
            dict: A dictionary where keys are the filenames (without the .json extension)
                  and values are the contents of the JSON files.
        """

        dataset = {}
        exisiting_files = [file_path for file_path in glob(f"{path}/*")]
        for file_path in exisiting_files:
            data_key = file_path.split("/")[-1].replace(".json", "")
            dataset[int(data_key)] = json.load(open(file_path, "r"))

        return dataset

    def save_dataset_as_files(self, dataset: dict, path: str) -> None:
        """
        Save the given dataset as individual JSON files in the specified directory.

        Args:
            dataset (dict): The dataset to be saved, where each key-value pair represents
                            a separate JSON file.
            path (str): The directory path where the JSON files will be saved. If the
                        directory does not exist, it will be created.

        Returns:
            None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for key in dataset:
            json.dump(dataset[key], open(f"{path}/{key}.json", "w"))
