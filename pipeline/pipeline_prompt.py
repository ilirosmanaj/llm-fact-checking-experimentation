import json

from langchain_core.messages import merge_message_runs

from pipeline.pipeline_base import PipelineBase
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from easydict import EasyDict as edict
from abc import abstractmethod
from typing import Any

class PipelinePrompt(PipelineBase):
    """
    A pipeline for any class using prompts.
    You should inherit from this class if you are using prompts in your class.
    """

    def __init__(self, config: dict):
        """
        Initializes the PipelinePrompt class. Here we does the following:
        - Load prompt file
        - Define prompt templates based on the message type and template.
        - Get message list templates

        Args:
            config (edict): Configuration file
        """
        super().__init__(config)
        self.prompts = edict(json.load(open(self.config.path.prompts)))
        self.prompt_templates = self.get_prompt_templates()
        self.merger = merge_message_runs()
        self.message_list_template = self.get_message_list_templates()

    def define_prompt_template(self, template_dict: dict, message_type: str):
        """
        Defines a prompt template based on the message type.

        Args:
            template_dict (dict): Dictionary containing template details.
            message_type (str): Type of the message, e.g., 'human' or 'system'.

        Returns:
            Any: An instance of the appropriate message prompt template.

        Raises:
            NotImplementedError: If the message type is not supported.
        """
        if message_type == "human":
            return HumanMessagePromptTemplate.from_template(template_dict["format"])
        elif message_type == "system":
            return SystemMessagePromptTemplate.from_template(template_dict["format"])
        else:
            raise NotImplementedError

    def get_prompt_templates(self):
        """
        Retrieves and constructs all prompt templates from the configuration.

        Returns:
            Dict[str, Any]: A dictionary where keys are template names and values are prompt templates.
        """
        prompt_templates = {}
        for message_type, template_dicts in self.prompts.items():
            for template_name, template_dict in template_dicts.items():
                prompt_templates[template_name] = self.define_prompt_template(
                    template_dict, message_type
                )
        return prompt_templates

    def get_message_list_templates(self) -> dict:
        """
        Generates a dictionary of message list templates for different purposes. message list templates are the list of system/human/ai messages

        all message lists should be defined here

        Returns:
            dict: A dictionary containing message list templates
        """
        message_list_template = {}
        for template_name, _ in self.prompts["human"].items():

            message_list_template[template_name] = (
                ChatPromptTemplate.from_messages(
                    [
                        self.prompt_templates[f"{template_name}_instruction"],
                        self.prompt_templates[template_name],
                    ]
                )
                | self.merger
            )
        return message_list_template

    @abstractmethod
    def get_model_prompt(self, **kwargs) -> Any:
        """
        Abstract method to be implemented in subclasses, defining how the model prompt is constructed.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The model prompt.
        """
        raise NotImplementedError
