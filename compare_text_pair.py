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
    experiment_manager.direct_text_match_test(
        config.inline_answer, config.inline_reference
    )
