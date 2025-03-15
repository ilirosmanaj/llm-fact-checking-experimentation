from dataset import *

"""
This script generates demonstration data using a specified model and dataset configuration.

Imports:
    from dataset import *: Imports all classes and functions from the dataset module.
    from model import *: Imports all classes and functions from the model module.
    from main import *: Imports all classes and functions from the main module.

Main Execution:
    - Initializes the model using the configuration specified in "config".
    - Creates an instance of "DemonstrationDataset" with the given configuration and model.
    - Determines the data path from the configuration.
    - Depending on the "demo_data_generation_method" specified in the configuration, it either:
        - Generates perfect scored generation samples.
        - Retrieves manual demonstration samples.
    - Iterates over the generated or retrieved data samples and generates demonstration data for each sample using the model's prompt template.

Configuration:
    - "config.demo_target_model": Specifies the target model for demonstration.
    - "config.model[config.demo_target_model].model_name": The name of the model to be used.
    - "config.demo_data_path": The path to the data used for generating demonstrations.
    - "config.demo_data_generation_method": The method used for generating demonstration data (e.g., "full_score" or "manual").
    - "config.save_data": A flag indicating whether to save the generated demonstration data.

Classes:
    - "DemonstrationDataset": A class that handles the creation and management of demonstration datasets.

Functions:
    - "generate_demo_data": Generates demonstration data for a given sample using the model's prompt template.
"""
from model import *
from main import *

if __name__ == "__main__":

    model = model_name_class_mapping[config.demo_target_model][
        config.model[config.demo_target_model].model_name
    ](config)
    demo_dataset = DemonstrationDataset(config, model=model)

    data_path = config.demo_data_path

    if config.demo_data_generation_method == "full_score":
        data_samples = demo_dataset.get_perfect_scored_generation_samples(data_path)
    elif config.demo_data_generation_method == "manual":
        data_samples = demo_dataset.get_manual_demo_samples(data_path)

    for idx, sample in enumerate(data_samples):
        example = demo_dataset.generate_demo_data(
            idx=idx,
            prompt_template=demo_dataset.model.get_model_prompt,
            save_data=config.save_data,
            **sample,
        )
