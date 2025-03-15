from dataset.base_dataset import *
from utils.utils import *
from model import *


class HallucinationDataset(BaseDataset):
    """
    HallucinationDataset is a class that handles the creation and retrieval of a hallucination dataset.
    The dataset is generated using the HallucinationDataGenerator and contains hallucinated answers
    along with their corresponding answer triplets.

    Attributes:
        hlcntn_data_generator (object): An instance of the hallucination data generator class.

    Methods:
        __init__(config: dict):
            Initializes the HallucinationDataset with the provided configuration.

        get_hlcntn_dataset() -> dict:
            Retrieves the hallucination dataset. If the dataset exists at the specified path, it loads and returns it.
            Otherwise, it creates the dataset, filters out passages with NaN values, and saves it for future use.

        create_hlcntn_dataset(save_path: str = None) -> dict:
            Creates the hallucination dataset by generating hallucinated data from the original dataset.
            Optionally saves the dataset to the specified path.

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
        super().__init__(config)
        self.logger = logger
        self.hlcntn_data_generator = model_name_class_mapping[
            "hallucination_data_generator"
        ][config.model.hallucination_data_generator.model_name](config, logger)

    def get_hlcntn_dataset(self):
        """
        Get the hallucination dataset.

        This method checks if the hallucination dataset is already present at the specified path.
        If the dataset exists, it loads and returns the dataset. If the dataset does not exist,
        it creates the dataset, filters out passages with NaN values, and saves the dataset to
        avoid repeated downloading and filtering in the future.

        Returns:
            dict: The hallucination dataset if it exists or is created successfully, otherwise an empty dictionary.
        """
        self.logger.info("==> Getting hallucination dataset")
        hlcntn_dataset_path = (
            f"{self.config.path.data.base}{self.config.path.data.hallucination_data}"
        )
        if os.path.exists(hlcntn_dataset_path):
            self.logger.info(f"==> Hallucination dataset exists locally, loading it")
            return self.load_files_as_dataset(hlcntn_dataset_path)
        else:
            self.logger.info("==> Hallucination dataset does not exist, creating it")
            if self.config.experiment_setup.save_all_triplets_as_dataset:
                return self.create_hlcntn_dataset(save_path=hlcntn_dataset_path)
            else:
                return {}

    def create_hlcntn_dataset(self, save_path=None):
        """
        Generates a hallucination dataset by using a triplet generator and the original dataset.

        Args:
            save_path (str, optional): The path where the generated dataset should be saved.
                                       If None, the dataset will not be saved to disk.

        Returns:
            dict: A dictionary where the keys are data indices and the values are the generated
                  hallucination data for each corresponding original data entry.
        """

        triplet_generator = model_name_class_mapping["triplet_generator"][
            self.config.model.triplet_generator.model_name
        ](self.config, self.logger)

        hlcntn_dataset = {
            data_idx: self.hlcntn_data_generator.generate_hlcntn_data_from_original_dataset(
                original_dataset=self.data_row_by_id(data_idx),
                triplet_generator=triplet_generator,
            )  # original data is in string format
            for data_idx in range(len(self.qa_dataset))
        }
        self.logger.info(f"==> Hallucination dataset created")
        self.logger.info(f"==> Number of Hallucination data: {len(hlcntn_dataset)}")
        if save_path is not None:
            self.logger.info(f"==> Saving Hallucination dataset")
            self.save_dataset_as_files(hlcntn_dataset, save_path)

        return hlcntn_dataset
