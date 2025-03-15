import json

from dataset.base_dataset import *
from datasets import load_dataset
from model import *


class BioASQDataset(BaseDataset):
    """
    BioSQDataset has three datasets
        1) corpus_text_dataset : corpus dataset at text format, used when generating corpus_triplet_dataset and adding context to question
        2) corpus_triplet_dataset : triplets generated from corpus_text_dataset and this is used at the fact checker. to create corpus_triplets, you need a triplet_generator
        3) qa_dataset : the question dataset with relevant passage
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.corpus_dataset = self.get_corpus_dataset()
        self.qa_dataset = self.get_qa_datset()
        self.corpus_triplets = self.get_corpus_triplet_dataset()

    def get_qa_datset(self) -> list:
        """
        Get the question answer dataset
        If the dataset is not present, create the dataset and filter the relevant_passage_ids and save the dataset.
        some passages have nan values and filtering is needed. And saving the dataset is needed to skip downloading and filtering data.
        """
        qa_dataset_path = (
            f"{self.config.path.data.base}{self.config.path.data.questions_answers}"
        )
        self.logger.info("==> Getting QA dataset")
        if self.config.experiment_setup.dataset == "all":
            keyword = ""
        else:
            keyword = self.config.experiment_setup.dataset
            qa_dataset_path = f"{qa_dataset_path.replace('.json','')}_{keyword}.json"

        if os.path.exists(qa_dataset_path):
            self.logger.info("==> QA dataset exists locally, loading it")
            qa_dataset = json.load(open(qa_dataset_path, "r"))

            self.logger.info(
                f"==> Number of QA Data for keyword '{keyword}': {len(qa_dataset)}"
            )
            return qa_dataset
        else:
            qa_dataset = self.create_qa_dataset(
                save_path=qa_dataset_path, keyword=keyword
            )
            return qa_dataset

    def create_qa_dataset(self, save_path=None, keyword="") -> list:
        """
        Creates a question-answer dataset filtered by a keyword and saves it to a specified path.

        Args:
            save_path (str, optional): The path where the dataset will be saved. If None, the dataset will not be saved. Defaults to None.
            keyword (str, optional): The keyword to filter questions. If empty, all questions are included. Defaults to "".

        Returns:
            list: A list of dictionaries, each containing 'id', 'question', 'answer', and 'relevant_passage_ids' keys.
        """
        self.logger.info("==> QA dataset does not exist, creating it")
        if self.config.experiment_setup.dataset == "all":
            keyword = ""
        else:
            keyword = self.config.experiment_setup.dataset
        qa_dataset = [
            {
                "id": qa_data["id"],
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "relevant_passage_ids": eval(
                    qa_data["relevant_passage_ids"]
                ),  # original data is in string format
            }
            for qa_data in load_dataset(
                "rag-datasets/rag-mini-bioasq", "question-answer-passages"
            )["test"]
            if keyword in qa_data["question"]
        ]
        qa_dataset = self.filter_relevant_passage_ids(qa_dataset)
        self.logger.info(
            f"==> Number of QA questions for keyword '{keyword}': {len(qa_dataset)}"
        )
        if save_path is not None:
            self.logger.info("==> Saving QA dataset")
            json.dump(qa_dataset, open(save_path, "w"))

        return qa_dataset

    def filter_relevant_passage_ids(self, qa_dataset):
        """
        Filters relevant passage IDs from the QA dataset.

        This method ensures that each passage ID in the QA dataset is also present in the corpus dataset.
        Passages with IDs not found in the corpus dataset are removed.
        Note: Passages with NaN values were already filtered out when creating the corpus dataset.

        Args:
            qa_dataset (list of dict): The QA dataset containing a list of dictionaries with relevant_passage_ids.

        Returns:
            list of dict: The updated QA dataset with filtered relevant_passage_ids.
        """
        qa_to_delete = []
        for idx, qa_data in enumerate(qa_dataset):
            qa_data["relevant_passage_ids"] = [
                passage_id
                for passage_id in qa_data["relevant_passage_ids"]
                if passage_id in self.corpus_dataset
            ]
            if len(qa_data["relevant_passage_ids"]) == 0:
                qa_to_delete.append(idx)
            qa_dataset[idx] = qa_data

        return qa_dataset

    def get_corpus_dataset(self):
        """
        Get the corpus dataset
        if the dataset is not present, create the dataset and filter the nan passages and save the dataset.
        some passage has nan values and filtering is needed. And saving the dataset is needed to skip downloading and filtering data.
        """
        self.logger.info("==> Getting corpus dataset")
        corpus_dataset_path = (
            f"{self.config.path.data.base}{self.config.path.data.corpus}"
        )
        if os.path.exists(corpus_dataset_path):
            self.logger.info("==> Corpus dataset exists, loading it")
            corpus_dataset = json.load(open(corpus_dataset_path, "r"))
            corpus_dataset = {int(key): value for key, value in corpus_dataset.items()}
            self.logger.info(
                f"==> Number of passages in the corpus data: {len(corpus_dataset)}"
            )
            return corpus_dataset
        else:
            self.logger.info("==> Corpus dataset does not exist, creating it")
            return self.create_corpus_dataset(save_path=corpus_dataset_path)

    def create_corpus_dataset(self, save_path=None) -> dict:
        """
        Creates a corpus dataset by loading data from the hugginaface "rag-datasets/rag-mini-bioasq" dataset,
        specifically the "text-corpus" split(the only split). It processes the passages by removing newline characters
        and ignoring passages that are "nan".

        Args:
            save_path (str, optional): The file path where the corpus dataset should be saved as a JSON file.
                                       If None, the dataset will not be saved to a file.

        Returns:
            dict: A dictionary where the keys are the IDs of the corpus data and the values are the processed passages.
        """
        corpus_dataset = {
            corpus_data["id"]: corpus_data["passage"].replace("\n", "")
            for corpus_data in load_dataset(
                "rag-datasets/rag-mini-bioasq", "text-corpus"
            )["passages"]
            if corpus_data["passage"] != "nan"
        }  # ignore all nan passages
        self.logger.info(
            f"==> Number of passages in the corpus dataset: {len(corpus_dataset)}"
        )

        if save_path is not None:
            self.logger.info("==> Saving corpus dataset")
            json.dump(corpus_dataset, open(save_path, "w"))

        return corpus_dataset

    def data_row_by_id(self, id: int) -> dict:
        """
        Retrieve a data row by its ID, including reference documents and reference triplets.

        Args:
            id (int or str): The unique identifier for the data row.

        Returns:
            dict: A dictionary containing the data row with the following keys:
                - All keys from the original data row in `self.qa_dataset`.
                - "reference_documents": A list of documents from `self.corpus_dataset`
                  corresponding to the relevant passage IDs.
                - "reference_triplets": Merged relevant reference triplets for the
                  relevant passage IDs.
        """

        data_row = {**self.qa_dataset[id]}
        data_row["reference_documents"] = [
            self.corpus_dataset[passage_id]
            for passage_id in data_row["relevant_passage_ids"]
        ]
        data_row["reference_triplets"] = self.merge_relevant_reference_triplets(
            data_row["relevant_passage_ids"]
        )
        return data_row

    def get_corpus_triplet_dataset(self) -> dict:
        """
        Retrieves the corpus triplet dataset.

        This method checks if the corpus triplet dataset exists at the specified path.
        If it exists, it loads and returns the dataset. If it does not exist, it checks
        the configuration to determine whether to create and save the dataset. If the
        configuration allows saving all triplets as a dataset, it creates the all dataset
        and saves it to the specified path. Otherwise, it returns an empty dictionary.

        Returns:
            dict: The corpus triplet dataset if it exists or is created, otherwise an empty dictionary.
        """
        self.logger.info("==> Getting corpus triplet dataset")
        corpus_triplet_dataset_path = f"{self.config.path.data.base}{self.config.path.data.corpus_triplet}_{self.config.experiment_setup.dataset}"
        if os.path.exists(corpus_triplet_dataset_path):
            self.logger.info("==> Corpus triplet dataset exists locally, loading it")
            return self.load_files_as_dataset(corpus_triplet_dataset_path)
        else:
            if self.config.experiment_setup.save_all_triplets_as_dataset:
                return self.create_corpus_triplet_dataset(
                    save_path=corpus_triplet_dataset_path
                )
            else:
                return {}

    def get_corpus_triplet_by_idx(self, passage_id: int, save_data=True) -> list:
        """
        Retrieves a corpus triplet by its passage ID. If the triplet is not already cached, it generates the triplet,
        caches it, and optionally saves it to disk.

        Args:
            passage_id (int or str): The ID of the passage for which to retrieve the triplet.
            save_data (bool, optional): Whether to save the generated triplet to disk. Defaults to True.

        Returns:
            list: The triplet associated with the given passage ID.
        """
        if passage_id in self.corpus_triplets:
            return self.corpus_triplets[passage_id]
        else:
            triplet_generator = model_name_class_mapping["triplet_generator"][
                self.config.model.triplet_generator.model_name
            ](
                self.config, self.logger
            )  # instantiate the triplet generator every time seems inefficient
            self.corpus_triplets[passage_id] = triplet_generator.forward(
                self.corpus_dataset[passage_id]
            )

            if save_data:
                save_path = f"{self.config.path.data.base}{self.config.path.data.corpus_triplet}_{self.config.experiment_setup.dataset}"
                triplet_dict_to_save = {passage_id: self.corpus_triplets[passage_id]}
                self.save_dataset_as_files(triplet_dict_to_save, save_path)
            return self.corpus_triplets[passage_id]

    def create_corpus_triplet_dataset(self, save_path: str) -> dict:
        """
        Generates a corpus triplet dataset and optionally saves it to a specified path.

        Args:
            save_path (str): The path where the generated corpus triplet dataset should be saved.
                             If None, the dataset will not be saved to disk.

        Returns:
            dict: A dictionary where keys are document IDs and values are the generated triplets.
        """
        self.logger.info(
            "==> Corpus triplet dataset does not exist, Creating corpus triplet dataset"
        )
        triplet_generator = model_name_class_mapping["triplet_generator"][
            self.config.model.triplet_generator.model_name
        ](self.config, self.logger)
        corpus_triplets = {
            document_id: triplet_generator.forward(passage)
            for document_id, passage in self.corpus_dataset.items()
        }
        self.logger.info(
            f"==> Number of Corpus Triplet Dataset: {len(corpus_triplets)}"
        )
        if save_path is not None:
            self.logger.info("==> Saving corpus triplet dataset")
            self.save_dataset_as_files(corpus_triplets, save_path)

        return corpus_triplets

    def merge_relevant_reference_triplets(self, passage_ids: list) -> list:
        """
        Merges relevant reference triplets for the given passage IDs.

        This method retrieves triplets from the corpus for each passage ID provided
        and merges them into a single list.

        Args:
            passage_ids (list): A list of passage IDs for which to retrieve and merge triplets.

        Returns:
            list: A list of triplets, where each triplet is represented as a list.
        """

        try:
            if self.config.model.fact_checker.split_reference_triplets:

                return self.get_segmented_triplets(passage_ids)

            else:
                return [
                    [
                        list(triplet)
                        for passage_id in passage_ids
                        for triplet in self.get_corpus_triplet_by_idx(
                            passage_id=passage_id, save_data=self.config.save_data
                        )
                    ]
                ]
        except Exception as e:
            self.logger.warning("==> Error in merging triplets: %s", str(e))
            self.logger.debug("==> Error passage_ids: %s", passage_ids)

            return []

    def get_segmented_triplets(self, passage_ids: list) -> list:
        """
        Get segmented triplets for the given passage IDs.

        This method retrieves triplets from the corpus for each passage ID provided
        and segments them into a list of segments, where each segment contains a list of triplets.

        Args:
            passage_ids (list): A list of passage IDs for which to retrieve and segment triplets.

        Returns:
            list: A list of segments, where each segment is a list of triplets.
        """
        all_segments = []
        current_segment = []
        max_length = self.config.model.fact_checker.max_reference_triplet_length
        for passage_id in passage_ids:
            triplets = self.get_corpus_triplet_by_idx(
                passage_id
            )  # Get triplet for the given passage_id
            self.logger.debug(f"==> Triplet idx/length: {passage_id},{len(triplets)}")
            if len(triplets) > max_length:
                self.logger.warning(
                    f"==> Triplet length of a single reference_document is greater than max_length : {passage_id}"
                )
                self.logger.debug(f"==> Triplet length: {len(triplets)}")
                self.logger.debug(f"==> Triplet: {triplets}")

            # Add triplets to the current segment if it does not exceed max_length
            if len(current_segment) + len(triplets) <= max_length:
                current_segment.extend(triplets)
            else:
                # If it exceeds, save the current segment and create a new one
                all_segments.append(current_segment)
                current_segment = triplets

        if len(current_segment) > 0:
            all_segments.append(current_segment)
        return all_segments
