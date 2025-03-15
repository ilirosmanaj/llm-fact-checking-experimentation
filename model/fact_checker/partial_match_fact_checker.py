from model.fact_checker.fact_checker import *


class PartialMatchFactChecker(FactChecker):
    """
    A fact checker that performs partial matching of triplets against a source dataset.

    Attributes:
        config (dict): Configuration dictionary for the fact checker.

    Methods:
        forward(data, source_triplets, threshold):
            Performs partial matching on the input data against the source triplets.

        check_partial_match_in_dataset(triplet, source_triplets, threshold):
            Checks for partial matches of a single triplet against the source triplets.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)

    def forward(self, data: list, source_triplets: list, threshold: int = 2):
        """
        Performs partial matching on the input data against the source triplets.
            Args:
                data (list): List of triplets to be checked.
                source_triplets (list): List of source triplets to match against.
                threshold (int, optional): Minimum number of matching elements in a triplet to consider it a match. Defaults to 2.
            Returns:
                tuple: A tuple containing two dictionaries:
                    - The first dictionary maps indices to boolean values indicating if a match was found.
                    - The second dictionary maps indices to the detailed match results.
        """
        source_triplets = self.flatten_triplets(source_triplets)

        match_results = {
            idx: self.check_partial_match_in_dataset(
                triplet, source_triplets, threshold=threshold
            )
            for idx, triplet in enumerate(data)
        }
        return {
            idx: bool(match_result) for idx, match_result in match_results.items()
        }, match_results

    def check_partial_match_in_dataset(
        self, triplet: list, source_triplets: list, threshold: int = 2
    ):
        """
        Checks for partial matches of a single triplet against the source triplets.
            Args:
                triplet (tuple): The triplet to be checked.
                source_triplets (list): List of source triplets to match against.
                threshold (int, optional): Minimum number of matching elements in a triplet to consider it a match. Defaults to 2.
            Returns:
                list: A list of tuples where each tuple contains a source triplet and the list of matching elements.

        """
        matches = [
            (
                source_triplet,
                [
                    triplet[i]
                    for i in range(len(triplet))
                    if triplet[i] == source_triplet[i]
                ],
            )
            for source_triplet in source_triplets
            if sum(triplet[i] == source_triplet[i] for i in range(len(triplet)))
            >= threshold
        ]
        return matches
