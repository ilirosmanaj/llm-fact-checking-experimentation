from dataset.hallucination_dataset import *
from dataset.bioasq_dataset import *
from  typing import Optional, Dict

class ExperimentDataset(BioASQDataset, HallucinationDataset):
    """
    ExperimentDataset class that inherits from BioASQDataset and HallucinationDataset.

    Attributes:
        hlcntn_dataset (dict): A dictionary containing hallucination data.

    Methods:
        __init__(config: dict):
            Initializes the ExperimentDataset with the given configuration, Get hallucination dataset(Note: hallucination dataset is loaded here not the hallucination data).

        get_dataset():
            Returns the QA dataset.

        get_corpus_triplets():
            Returns the corpus triplets.

        hlcntn_data_row_by_id(id, save_data=True):
            Retrieves a hallucination data row by its ID. If the data does not exist, it generates the data,
            saves it if specified, and then returns it.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.logger = logger
        BioASQDataset.__init__(self, config, logger)
        HallucinationDataset.__init__(self, config, logger)
        self.hlcntn_dataset = self.get_hlcntn_dataset()
        # Graph can be represented as set of triplets
        # self.corpus_graph = self.get_corpus_graph(self.corpus_triplets)

    def get_dataset(self):
        return self.qa_dataset

    def get_corpus_triplets(self):
        return self.corpus_triplets

    def hlcntn_data_row_by_id(self, idx: int, save_data=True) -> Optional[Dict]:
        """
        Retrieve or generate hallucination data for a given ID.

        This method checks if the hallucination data for the given ID is already present in the `hlcntn_dataset`.
        If it is, the data is returned. If not, it generates the hallucination data using the original dataset
        and a triplet generator, saves it if required, and then returns the generated data.

        Args:
            id (int): The unique identifier for the data row.
            save_data (bool, optional): Flag indicating whether to save the generated hallucination data. Defaults to True.

        Returns:
            Optional[Dict]: The hallucination data corresponding to the given ID, or None if an error occurs.
        """

        if idx in self.hlcntn_dataset:
            return self.hlcntn_dataset[idx]
        else:
            org_data = self.data_row_by_id(idx)
            triplet_generator = model_name_class_mapping["triplet_generator"][
                self.config.model.triplet_generator.model_name
            ](self.config, self.logger)
            try:
                hlcntn_data = self.hlcntn_data_generator.generate_hlcntn_data_from_original_dataset(
                    original_dataset=org_data, triplet_generator=triplet_generator
                )
                self.hlcntn_dataset[idx] = hlcntn_data
                if save_data:
                    hlcntn_dataset_path = f"{self.config.path.data.base}{self.config.path.data.hallucination_data}"
                    triplet_dict_to_save = {idx: hlcntn_data}
                    self.save_dataset_as_files(
                        triplet_dict_to_save, hlcntn_dataset_path
                    )
                return self.hlcntn_dataset[idx]
            except Exception as e:
                self.logger.warning(
                    f"Error generating hallucination data for ID {id}: {e}"
                )
                return None
