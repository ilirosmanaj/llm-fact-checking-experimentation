import argparse
import json
from easydict import EasyDict as edict

"""
Main is just for configurations
"""


def config_parser(args):
    config_path = f"config.json"
    with open(config_path, "rb") as f:
        config = dict(json.load(f))

    config = edict({**config, **{k: v for k, v in vars(args).items() if v is not None}})
    config = override_experiment_path(config)
    return config


def args2dict(config_file=None):
    """
    Parses command-line arguments and returns them as a dictionary.

    Args:
        config_file (str, optional): Default configuration file path. Defaults to None.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.

    Command-line Arguments:
        -c, --config (str): Path to the configuration file.
        -a, --answer_generator (str): Path to the answer generator.
        -t, --triplet_generator (str): Path to the triplet generator.
        -s, --fact_checker (str): Path to the fact checker.
        -e, --experiment_name (str, required): The name of the experiment.
        -sr, --save_result (bool): Whether to save results.
        -sd, --save_data (bool): Whether to save data.
        -hl, --evalute_hlcntn (bool): Whether to evaluate with a certain option (hlcntn).
        -rp, --do_reprompt (bool): Whether to perform a re-prompt procedure.
        --num_test_samples (int): Number of test samples.
        --sample_idx (int): A specific sample index to use.
        --demo_target_model (str): Path or name of the target model for demo. used only in generate_demonstrations.py .
        --demo_data_path (str): Path to demo data.  used only in generate_demonstrations.py .
        --demo_data_generation_method (str): How demo data is generated.  used only in generate_demonstrations.py .
    """
    args = argparse.ArgumentParser(description="experiment")
    args.add_argument("-c", "--config", default=config_file, type=str)
    args.add_argument("-a", "--answer_generator", default=None, type=str)
    args.add_argument("-t", "--triplet_generator", default=None, type=str)
    args.add_argument("-s", "--fact_checker", default=None, type=str)
    args.add_argument("-e", "--experiment_name", required=True, type=str)
    args.add_argument("-sr", "--save_result", default=True, type=bool)
    args.add_argument("-sd", "--save_data", default=True, type=bool)
    args.add_argument("-hl", "--evalute_hlcntn", default=True, type=bool)
    args.add_argument("-rp", "--do_reprompt", default=False, type=bool)
    args.add_argument("-l", "--logger_level", default="INFO", type=str)
    args.add_argument("--num_test_samples", default=None, type=int)
    args.add_argument("--sample_idx", default=None, type=int)
    args.add_argument("--demo_target_model", default=None, type=str)
    args.add_argument("--demo_data_path", default=None, type=str)
    args.add_argument("--demo_data_generation_method", default=None, type=str)
    args.add_argument("--inline_answer", default=None, type=str)
    args.add_argument("--inline_reference", default=None, type=str)
    config_dict = config_parser(args.parse_args())
    return config_dict


def override_experiment_path(config):
    if hasattr(config, "sample_idx"):
        config.experiment_name = f"{config.experiment_name}_{config.sample_idx}"
    return config


config = args2dict(config_file="training")
