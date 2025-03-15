from dataset import *
from rag.llm_fact_checking_system import *
from easydict import EasyDict as edict
import json

class ExperimentManager:
    """
    ExperimentManager is a temporal name for this class, will figure out a better name later.

    ExperimentManager class is responsible for managing the experiments.

    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.model = LLMFactCheckingSystem(config, logger)
        self.dataset = ExperimentDataset(config, logger)
        self.logger = logger

    def run_experiment(self, save_result=True, evalute_hlcntn=True, do_reprompt=False):
        """
        Run the experiment by iterating over the dataset and making predictions.

        Args:
            save_result (bool): Whether to save the experiment results. Default is True.
            evalute_hlcntn (bool): Whether to evaluate the hlcntn dataset. Default is True.
            do_reprompt (bool): Whether to perform reprompting if the precision is below a threshold. Default is False.

        Returns:
            tuple: A tuple containing:
                - metrics (dict): The calculated precision statistics for the experiment.
                - hlcntn_metrics (dict or None): The evaluation metrics for the hlcntn dataset if evalute_hlcntn is True, otherwise None.

        Notes:
            - The function iterates over the dataset and makes predictions using the model.
            - If "do_reprompt" is True and the precision is below a threshold, reprompting is performed.
            - The results and metrics are saved if "save_result" is True.
            - The function prints the start and end of the experiment.
        """

        prediction_result = []
        if "num_test_samples" in self.config:
            num_samples = self.config.num_test_samples
        else:
            num_samples = len(self.dataset.qa_dataset)

        self.logger.info(
            f"Experiment settings are num_samples: {num_samples}, save: {save_result}, evalute_hlcntn: {evalute_hlcntn}, do_reprompt: {do_reprompt}"
        )
        self.logger.debug(f"All Experiment config: {self.config}")
        self.logger.info(
            "==================================== Experiment started ======================================="
        )
        for idx in range(num_samples):
            if hasattr(self.config, "sample_idx"):
                if idx != self.config.sample_idx:
                    continue

            self.logger.info(f"=== Current question index: {idx + 1} ")

            question_data, output = self.evaluate_non_hlcntn_sample(idx)

            if output is None:
                continue
            output.update(
                {
                    "precision": (
                        sum(
                            [
                                i
                                for i in list(
                                    output["fact_check_prediction_binary"].values()
                                )
                                if type(i) == bool
                            ]
                        )
                        / len(output["fact_check_prediction_binary"])
                    ),
                    "question": question_data["question"],
                    "idx": idx,
                    "reference_documents": question_data["reference_documents"],
                    "reference_triplets": question_data["reference_triplets"],
                }
            )
            prediction_result.append(output)

            metrics = self.calculate_precision_stats(prediction_result)
            if save_result:
                self.save_experiment_result(
                    metrics,
                    prediction_result,
                    config=self.config,
                    result_type="original",
                )
            tabs = "\t\t\t\t\t\t\t\t\t"
            answer_triplet_string = "".join(
                [
                    f"\n{tabs}   - {idx}: {triplet}"
                    for idx, triplet in enumerate(output["answer_triplets"])
                ]
            )
            self.logger.info(
                f"{tabs} Question: {question_data['question']}"
                f"\n{tabs} Answer: {output['generated_answer']} "
                f"\n{tabs} Reference documents: {output['reference_documents']} "
                f"\n{tabs} Number of answer triplets: {len(output['answer_triplets'])} "
                f"\n{tabs} Answer triplets: {answer_triplet_string} "
                f"\n{tabs} Fact checking output: {output['fact_check_prediction_binary']}"
            )
            self.logger.info(f"==> Current precision: {metrics['precision']}")
            self.logger.info(
                f"==> Current non-hallucinated triplets (predicted as true/all): {metrics['num_non_hlcntn_triplets_correctly_predicted']}/{metrics['num_non_hlcntn_triplets']}"
            )
            self.logger.info(
                "================================================================================================"
            )
        self.logger.info(
            "==================================== Experiment ended =========================================="
        )

        if evalute_hlcntn:
            hlcntn_metrics = self.evaluate_hlcntn_dataset(
                save_result=save_result,
                do_reprompt=do_reprompt,
                num_samples=num_samples,
            )
        else:
            hlcntn_metrics = None
        return metrics, hlcntn_metrics

    def evaluate_non_hlcntn_sample(self, idx, retry_num=0):

        question_data = self.dataset.data_row_by_id(idx)

        # todo : change filtering logic to a method
        if retry_num >= self.config.experiment_setup.system_retry:
            self.logger.warning(
                "=============================================================="
            )
            return question_data, None

        output = self.model.forward(question_data)

        if len(output["fact_check_prediction_binary"]) == 0:
            self.logger.warning(
                f"No fact checking could be extracted for: '{question_data['question']}'. Skipping it."
            )
            self.logger.debug(f"Empty fact check prediction: {output}")

            self.logger.warning("==>Retrying")
            return self.evaluate_non_hlcntn_sample(idx, retry_num + 1)

        elif len(output["answer_triplets"]) != len(
            output["fact_check_prediction_binary"]
        ):
            self.logger.error(
                f"Number of predictions doesn't match the one for triplets for {question_data['question']}. Skipping"
            )
            self.logger.debug(f"Length mismatch output: {output}")

            self.logger.warning("==>Retrying")
            return self.evaluate_non_hlcntn_sample(idx, retry_num + 1)

        elif any([i == "" for i in output["answer_triplets"]]):
            self.logger.warning(
                f"Empty triplets exist in the answer of the question: '{question_data['question']}'. Skipping it."
            )
            self.logger.debug(f"Empty answer triplets: {output}")

            self.logger.warning("==>Retrying")
            return self.evaluate_non_hlcntn_sample(idx, retry_num + 1)

        elif any([i == "" for i in question_data["reference_triplets"]]):
            self.logger.warning(
                f"Empty triplets exist in the reference passages of the question: '{question_data['question']}'. Skipping it."
            )
            self.logger.debug(f"Empty reference triplets: {question_data}")

            self.logger.warning("==>Retrying")
            return self.evaluate_non_hlcntn_sample(idx, retry_num + 1)

        if (
            output["generated_answer"].startswith("There is no evidence")
            and len(output["fact_check_prediction_binary"]) == 1
        ):
            self.logger.warning(
                f"Answer generator found no evidence for question: '{question_data['question']}'. Skipping it."
            )
            self.logger.debug(f"No evidence answer: {output}")

            self.logger.warning("==>Retrying")
            return self.evaluate_non_hlcntn_sample(idx, retry_num + 1)

        return question_data, output

    def evaluate_hlcntn_dataset(
        self, save_result=True, do_reprompt=False, num_samples=None
    ):
        """
        Evaluate the hallucination dataset.

        Parameters:
        - save_result (bool): Whether to save the experiment results. Default is True.
        - do_reprompt (bool): Whether to perform reprompting if precision is below a threshold. Default is False.
        - num_samples (int, optional): Number of samples to evaluate. If not provided, it will use the number of test samples from the config or the length of the dataset.

        Returns:
        - dict: A dictionary containing the calculated metrics for the hallucination experiment.

        The function iterates over the dataset, processes each data row, and evaluates the model's performance on hallucination detection. It calculates precision and other relevant metrics, optionally performs reprompting, and saves the results if specified.

        """
        if "num_test_samples" in self.config:
            num_samples = self.config.num_test_samples
        else:
            num_samples = len(self.dataset.qa_dataset)
        prediction_result = []

        # dummy splitter
        self.logger.info("")
        self.logger.info("")
        self.logger.info("")

        self.logger.info(
            "================================= Hallucination experiment started ============================="
        )
        for idx in range(num_samples):

            if hasattr(self.config, "sample_idx"):
                if idx != self.config.sample_idx:
                    continue

            self.logger.info(f"=== Current question index: {idx + 1} ")

            data, hlcntn_data, output = self.evaluate_hlcntn_sample(idx)
            if output is None:
                continue

            output.update(
                {
                    "precision": sum(
                        list(output["fact_check_prediction_binary"].values())
                    )
                    / len(output["fact_check_prediction_binary"]),
                    "reference_documents": data["reference_documents"],
                    "reference_triplets": data["reference_triplets"],
                    "question": data["question"],
                    "idx": idx,
                    "generated_answer": hlcntn_data["generated_answer"],
                    "generated_non_hlcntn_answer": hlcntn_data[
                        "generated_non_hlcntn_answer"
                    ],
                    "generated_hlcntn_answer": hlcntn_data["generated_hlcntn_answer"],
                    "non_hlcntn_triplets": hlcntn_data["non_hlcntn_triplets"],
                    "hlcntn_triplets": [
                        triplet
                        for idx, triplet in enumerate(hlcntn_data["answer_triplets"])
                        if hlcntn_data["hlcntn_triplet_index"][idx]
                    ],
                    "hlcntn_triplet_index": hlcntn_data["hlcntn_triplet_index"],
                }
            )
            if (
                do_reprompt
            ):  # todo : think reprompter should be elsewhere. this is experiment pipeline but repropter looks more related to model not experiment
                if output["precision"] < self.config.model.reprompter.threshold:
                    reprompt_output = self.model.reprompter_forward(data, output)
                    if (
                        len(reprompt_output["reprompt_fact_check_prediction_binary"])
                        > 0
                    ):  # exception
                        output.update(reprompt_output)
                        output.update(
                            {
                                "reprompt_precision": sum(
                                    list(
                                        output[
                                            "reprompt_fact_check_prediction_binary"
                                        ].values()
                                    )
                                )
                                / len(output["reprompt_fact_check_prediction_binary"])
                            }
                        )
            prediction_result.append(output)

            hlcntn_metrics = self.calculate_precision_stats(prediction_result)
            hlcntn_metrics.update(self.calculate_hlcntn_metrics(prediction_result))

            if save_result:
                self.save_experiment_result(
                    hlcntn_metrics,
                    prediction_result,
                    config=self.config,
                    result_type="hlcntn",
                )

            tabs = "\t\t\t\t\t\t\t\t\t"
            hallucination_indexes_as_dict = {
                idx: value for idx, value in enumerate(output["hlcntn_triplet_index"])
            }
            answer_triplet_string = "".join(
                [
                    f"\n{tabs}   - {idx}: {triplet}"
                    for idx, triplet in enumerate(output["answer_triplets"])
                ]
            )
            self.logger.info(
                f"{tabs} Question: {data['question']}"
                f"\n{tabs} Non-hallucinated Answer: {output['generated_non_hlcntn_answer'].strip()}"
                f"\n{tabs} Hallucinated Answer: {output['generated_hlcntn_answer'].strip()} "
                f"\n{tabs} Reference documents: {output['reference_documents']} "
                f"\n{tabs} Number of answer triplets: {len(output['hlcntn_triplets'])}"
                f"\n{tabs} Number of hallucinated triplets: {len([1 for o in output['hlcntn_triplet_index'] if o is True])}"
                f"\n{tabs} Hallucinated answer triplets: {output['hlcntn_triplets']}"
                f"\n{tabs} Answer triplets: {answer_triplet_string}"
                f"\n{tabs} Hallucinated indexes: {hallucination_indexes_as_dict}"
                f"\n{tabs} Fact checking output: {output['fact_check_prediction_binary']}"
            )
            self.logger.info(f"==> Current precision: {hlcntn_metrics['precision']}")
            self.logger.info(
                f"==> Current specificity: {hlcntn_metrics['specificity']}"
            )
            self.logger.info(
                f"==> Current non-hallucinated triplets (correctly predicted as true/all): {hlcntn_metrics['num_non_hlcntn_triplets_correctly_predicted']}/{hlcntn_metrics['num_non_hlcntn_triplets']}"
            )
            self.logger.info(
                f"==> Current hallucinated triplets (correctly predicted as hallucinations/all): {hlcntn_metrics['num_hlcntn_triplets_correctly_predicted']}/{hlcntn_metrics['num_hlcntn_triplets']}"
            )
            self.logger.info(
                "========================================================================================="
            )
        self.logger.info(
            "================================= Hallucination experiment ended ========================"
        )
        return hlcntn_metrics

    def evaluate_hlcntn_sample(self, idx, retry_num=0):
        #####
        data = self.dataset.data_row_by_id(idx)
        hlcntn_data = self.dataset.hlcntn_data_row_by_id(
            idx, save_data=self.config.save_data
        )  # change here

        # failed in parsing
        if retry_num >= self.config.experiment_setup.system_retry:
            self.logger.warning(
                "=============================================================="
            )
            return data, hlcntn_data, None

        if hlcntn_data is None:
            self.logger.warning(
                f"Failed to create hallucinated version for {data['question']}. Skipping"
            )

            self.logger.warning("==>Retrying")
            return self.evaluate_hlcntn_sample(idx, retry_num + 1)

        output = self.model.hlcntn_forward(data, hlcntn_data)
        if len(output["fact_check_prediction_binary"]) != len(
            hlcntn_data["hlcntn_triplet_index"]
        ):
            self.logger.warning(
                f"Number of predictions doesn't match the one for triplets for {data['question']}. Skipping"
            )
            self.logger.debug(f"Length mismatch output: {output}")
            self.logger.debug(f"Length mismatch hlcntn_data: {hlcntn_data}")

            self.logger.warning("==>Retrying")
            return self.evaluate_hlcntn_sample(idx, retry_num + 1)

        return data, hlcntn_data, output

    def calculate_precision_stats(self, prediction_result: list):
        """˝
        Calculate precision statistics from prediction results. If

        Args:
            prediction_result (list of dict): A list of dictionaries where each dictionary contains:
                - "precision" (float): The precision score of each the fact check result.
                - "reference_triplets" (list): A list of reference triplets.
                - "generated_answer" (str): The generated answer. - this could be artificially generated hallucination anwer or just generated answer
                - "reprompt_precision" (float, optional): The precision score after reprompting.

        Returns:
            dict: A dictionary containing:
                - "precision" (float): precision score or the experiment. This is calculated by number_of_corect_predictions_for_non_hallucination / total_number_of_predictions_for_non_hallucination
                - "avg_reprompt_score" (float or None): The average reprompt precision score, or None if not applicable.
                - "std_reprompt_score" (float or None): The standard deviation of the reprompt precision scores, or None if not applicable.
                - "avg_reprompt_improvement" (float or None): The average improvement in precision after reprompting, or None if not applicable.
                - "std_reprompt_improvement" (float or None): The standard deviation of the improvement in precision after reprompting, or None if not applicable.
                - "num_triplets" (int): The number of triplets in the experiment.
                - "num_non_hlcntn_triplets_correctly_predicted" (int): The number of correct predictions for non-hallucination triplets.
        """

        model_predictions = []
        for item in prediction_result:
            # exclude hallucination triplets for hallucinated experiments
            if "hlcntn_triplet_index" in item:
                model_predictions.extend(
                    [
                        value
                        for idx, value in item["fact_check_prediction_binary"].items()
                        if item["hlcntn_triplet_index"][idx] == False
                    ]
                )
            else:
                model_predictions.extend(
                    [
                        value
                        for idx, value in item["fact_check_prediction_binary"].items()
                    ]
                )

        precision = sum(model_predictions) / len(
            model_predictions
        )  # only for non-hallucination
        num_correct_predictions = sum(model_predictions)

        precisions = [
            item["precision"]
            for item in prediction_result
            if all([len(i) == 3 for i in item["reference_triplets"]])
            and item["generated_answer"] != ""
        ]
        reprompt_score = [
            item["reprompt_precision"] if "reprompt_precision" in item else None
            for item in prediction_result
        ]
        reprompt_score_without_none = [
            item for item in reprompt_score if item is not None
        ]
        if len(reprompt_score_without_none) > 1:
            avg_reprompt_score = sum(reprompt_score_without_none) / len(
                reprompt_score_without_none
            )
            std_reprompt_score = np.std(reprompt_score_without_none)
            # reprompt_improvement
            reprompt_improvement = [
                reprompt_score[i] - precisions[i]
                for i in range(len(precisions))
                if reprompt_score[i] is not None
            ]
            avg_reprompt_improvement = sum(reprompt_improvement) / len(
                reprompt_improvement
            )
            std_reprompt_improvement = np.std(reprompt_improvement)
        else:
            avg_reprompt_score = None
            std_reprompt_score = None
            avg_reprompt_improvement = None
            std_reprompt_improvement = None
        return {
            "precision": precision,
            "avg_reprompt_score": avg_reprompt_score,
            "std_reprompt_score": std_reprompt_score,
            "avg_reprompt_improvement": avg_reprompt_improvement,
            "std_reprompt_improvement": std_reprompt_improvement,
            "num_non_hlcntn_triplets": len(model_predictions),
            "num_non_hlcntn_triplets_correctly_predicted": num_correct_predictions,
        }

    def calculate_hlcntn_metrics(self, result: list):
        """
        Calculate HLCTN (Hallucination Containment) metrics from the given result.

        This function computes the specificity for the HLCTN metrics
        based on the model predictions and ground truth values provided in the result.

        Args:
            result (list): A list of dictionaries containing the model predictions and ground truth values.
                Each dictionary should have the following keys:
                - "fact_check_prediction_binary" (dict): A dictionary where the keys are indices and the values are binary predictions.
                - "hlcntn_triplet_index" (list): A list of ground truth binary values indicating the presence of hallucinations.

        Returns:
            dict: A dictionary containing the following HLCTN metrics:
                - "specificity" (float): The specificity score, tn/(fp+tn).
                - "num_hlcntn_triplets" (int): The number of hallucination triplets.
                - "num_hlcntn_triplets_correctly_predicted" (int): The number of correct hallucination samples.
        """
        model_predictions = []
        ground_truth = []
        for item in result:
            model_predictions.extend(
                [value for idx, value in item["fact_check_prediction_binary"].items()]
            )
            ground_truth.extend(item["hlcntn_triplet_index"])

        recall, num_hlcntn_triplets_correctly_predicted = specificity_and_samples(
            [i == False for i in ground_truth], model_predictions
        )
        return {
            "specificity": recall,
            "num_hlcntn_triplets": len([i for i in ground_truth if i == True]),
            "num_hlcntn_triplets_correctly_predicted": num_hlcntn_triplets_correctly_predicted,
        }

    def save_experiment_result(
        self,
        metrics: dict,
        prediction_result: list,
        config: edict,
        result_type="original",
    ):
        """
        Save the results of an experiment to the specified directory.

        Parameters:
        - metrics (dict): A dictionary containing the metrics of the experiment.
        - prediction_result (list): A list of dictionaries containing the prediction results.
        - config (object): Configuration object containing experiment settings.
        - result_type (str, optional): Type of result to save. Defaults to "original".
          Can be "original" or "hlcntn" (hallucination).

        The function performs the following steps:
        1. Constructs the path to save the experiment results based on the experiment name.
        2. Filters out prediction results that do not contain any answer triplets.
        3. Creates the experiment result directory if it does not exist.
        4. Saves the configuration, prompt bank, and commit information if the directory is newly created.
        5. Determines the paths for saving metrics and predictions based on the result type.
        6. Saves the metrics and prediction results to their respective files.
        """
        experiment_result_path = (
            f"{self.config.path.experiment_result.base}{config.experiment_name}/"
        )
        # temporal code, delete if there are no source triplets
        prediction_result = [
            i for i in prediction_result if len(i["answer_triplets"]) != 0
        ]

        if not os.path.exists(experiment_result_path):
            # create directory
            os.makedirs(experiment_result_path)
            # save config
            json.dump(config, open(f"{experiment_result_path}/config.json", "w"))
            # save prompt bank
            json.dump(
                json.load(open(self.config.path.prompts)),
                open(f"{experiment_result_path}/prompt_bank.json", "w"),
            )
            # save, commit hash and message
            commit_info = get_current_commit_hash_and_message()
            json.dump(
                commit_info, open(f"{experiment_result_path}/commit_info.json", "w")
            )
        else:
            pass

        if result_type == "original":
            metrics_path = (
                f"{experiment_result_path}{self.config.path.experiment_result.metrics}"
            )
            predictions_path = f"{experiment_result_path}{self.config.path.experiment_result.predictions}"

        elif result_type == "hlcntn":
            metrics_path = f"{experiment_result_path}{self.config.path.experiment_result.hallucination_metrics}"
            predictions_path = f"{experiment_result_path}{self.config.path.experiment_result.hallucination_predictions}"

        # Save results to file
        json.dump(metrics, open(metrics_path, "w"))
        json.dump(prediction_result, open(predictions_path, "w"))

    def save_hlcntn_experiment_result(self, metrics: dict, prediction_result: dict):
        """
        Save hallucination experiment results to specified file paths.

        Args:
            metrics (dict): A dictionary containing the metrics of the experiment.
            prediction_result (dict): A dictionary containing the prediction results of the experiment.

        Saves:
            The metrics and prediction results to their respective file paths defined in the configuration.
        """
        metrics_path = f"{self.config.path.experiment_result.base}{self.config.path.experiment_result.hallucination_metrics}"
        predictions_path = f"{self.config.path.experiment_result.base}{self.config.path.experiment_result.hallucination_predictions}"

        # Save results to file
        json.dump(metrics, open(metrics_path, "w"))
        json.dump(prediction_result, open(predictions_path, "w"))

    def direct_text_match_test(self, anwer_text, reference_text):
        """
        Perform forward pass for direct text matching.

        Args:
            answer_text (str): The text to be checked.
            reference_text (str): The reference text to compare against.

        Returns:
            dict: A dictionary containing the following:
                - "answer_triplets" (list): The triplets extracted from the answer text.
                - "fact_check_prediction_binary" (dict): The binary prediction result of the fact checker.
                - "false_triplet_index" (int): The index of the triplet predicted as False.
        """
        assert hasattr(self.config, "inline_answer")
        assert hasattr(self.config, "inline_reference")

        result = self.model.direct_text_match_forward(anwer_text, reference_text)

        tabs = "\t\t\t\t\t\t\t\t\t"

        self.logger.info(
            f"\n{tabs} Answer: {anwer_text}"
            f"\n{tabs} Reference: {reference_text}"
            f"\n{tabs} Answer triplets: {result['answer_triplets']}"
            f"\n{tabs} Reference triplets: {result['reference_triplets']}"
            f"\n{tabs} Fact checking output: {result['fact_check_prediction_binary']}"
        )


def test():
    config = edict(json.load(open("config.json", "r")))
    experiment_manager = ExperimentManager(config)
    metrics = experiment_manager.run_experiment(save_result=True)
    print(metrics)
