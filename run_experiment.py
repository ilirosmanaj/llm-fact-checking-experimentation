import warnings
import urllib3

warnings.simplefilter("ignore", category=urllib3.exceptions.HTTPWarning)
warnings.simplefilter("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

from dotenv import load_dotenv

from experiment_manager import ExperimentManager
from main import *
from utils.utils import ExperimentLogger

load_dotenv()

if __name__ == "__main__":
    logger = ExperimentLogger(
        "",
        log_path=f"{config.path.experiment_result.base}{config.experiment_name}/",
        logger_level=config.logger_level,
    )
    experiment_manager = ExperimentManager(config, logger)
    experiment_manager.run_experiment(
        save_result=config.save_result,
        evalute_hlcntn=config.evalute_hlcntn,
        do_reprompt=config.do_reprompt,
    )
