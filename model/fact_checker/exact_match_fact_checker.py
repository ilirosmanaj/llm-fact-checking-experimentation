from model.fact_checker.fact_checker import *


class ExactMatchFactChecker(FactChecker):
    """
    ExactMatchFactChecker is a class that checks if a given triplet exists in a source dataset using exact match.

    Methods
    __init__(config: dict)
        Initializes the ExactMatchFactChecker with the given configuration.

    forward(data: List[List], reference_triplets: List[List]) -> Tuple[Dict[int, bool], None]
        Checks each triplet in the data to see if it exists in the reference_triplets dataset and returns a dictionary
        with the results.

    check_triplet_exists_in_dataset(triplet: List, dataset: List[List]) -> bool
        Checks if a given triplet exists in the dataset.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)

    def forward(
        self,
        answer_triplets: List[List],
        reference_triplets: List[List],
    ):
        """
        Perform a forward pass to check if each triplet in the input data exists in the source triplets dataset.

        Args:
            answer_triplets (List[List]): A list of triplets to be checked.
            reference_triplets (List[List]): A list of triplets representing the source dataset.

        Returns:
            Tuple[Dict[int, bool], None]: A dictionary where the keys are the indices of the input triplets and the values are booleans indicating whether each triplet exists in the source dataset, and None.
        """
        reference_triplets = self.flatten_triplets(reference_triplets)

        fact_check_prediction_binary = {
            idx: self.check_triplet_exists_in_dataset(
                triplet=triplet, dataset=reference_triplets
            )
            for idx, triplet in enumerate(answer_triplets)
        }
        return fact_check_prediction_binary, None

    def check_triplet_exists_in_dataset(self, triplet, dataset):
        """
        Checks if a given triplet exists in the dataset.

        Args:
            triplet (tuple): The triplet to check for existence in the dataset.
            dataset (list or set): The dataset in which to search for the triplet.

        Returns:
            bool: True if the triplet exists in the dataset, False otherwise.
        """

        return triplet in dataset  # can be changed to other search methods
