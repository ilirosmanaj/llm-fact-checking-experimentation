from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import logging
import dotenv
import os

from git import Repo

# Load environment variables from a .env file
dotenv.load_dotenv()
# Retrieve the local repository path from an environment variable
repo_path = os.getenv("REPO_PATH")


def get_current_commit_hash_and_message():
    """
    Get the current commit hash and message of the repository.

    Returns:
        str: The current commit hash
        str: The commit message
    """
    # Initialize a Repo object pointing to the given repository path
    repo = Repo(repo_path)
    # Return a dictionary containing the latest commit's hash and message
    return {"hash": repo.head.commit.hexsha, "message": repo.head.commit.message}


def compute_false_omission_rate(y_true, y_pred):
    """
    Compute the False Omission Rate (FOR) from true labels and predicted labels.

    Parameters:
    - y_true (list or array-like): True labels. - this means the triplet is based on gold triplets
    - y_pred (list or array-like): Predicted labels. - this means the triplet is not based on gold triplets

    Returns:
    - float: False Omission Rate (FOR).
    """
    # Compute the confusion matrix: tn (true negatives), fp (false positives),
    # fn (false negatives), tp (true positives)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # FOR = fn / (fn + tn), but handle the case where denominator could be zero
    return fn / (fn + tn) if (fn + tn) > 0 else 0


def specificity_and_samples(y_true, y_pred):
    """
    Compute the recall metric that we aim with true labels and predicted labels. here, True means the non-hallucinated triplets and False means the hallucinated triplets.

    Parameters:
    - y_true (list or array-like): True labels. - this means the triplet is based on gold triplets
    - y_pred (list or array-like): Predicted labels. - this means the triplet is not based on gold triplets

    Returns:
    - float: hallucination recall that we aimed for.
    """
    # Compute the confusion matrix: tn (true negatives), fp (false positives),
    # fn (false negatives), tp (true positives)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # FOR = fn / (fn + tn), but handle the case where denominator could be zero
    return tn / (fp + tn) if (fp + tn) > 0 else 0, int(tn)


class ExperimentLogger(logging.Logger):
    def __init__(self, name, log_path: str, logger_level="INFO"):
        super().__init__("")
        self.log_path = log_path
        self.configure_logger(logger_level=logger_level)

    # def set_name(self):
    #     pass
    def configure_logger(self, logger_level="INFO"):
        if logger_level == "DEBUG":
            logger_level = logging.DEBUG
        elif logger_level == "INFO":
            logger_level = logging.INFO
        else:
            logger_level = getattr(logging, logger_level)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        log_file_name = self.log_path + "log.txt"

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file_name)

        file_handler.setLevel(logger_level)
        file_handler.setFormatter(formatter)

        self.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
