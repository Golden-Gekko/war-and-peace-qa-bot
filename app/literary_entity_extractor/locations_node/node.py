import re
import os
from typing import Any, Dict, Type
import yaml

from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from .model import Locations
from ..states import CreatorState
from ..parsers import ThinkAwarePydanticOutputParser


class LocationsNode():
    def __init__(
        self,
        llm,
        parser: Type[PydanticOutputParser] = ThinkAwarePydanticOutputParser,
        model: Type[BaseModel] = Locations,
    ):
        self.llm = llm
        self.parser = parser(pydantic_object=model)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        promt_path = os.path.join(current_dir, 'promt.yaml')
        self.prompt = self._load_prompt(promt_path)
        self.chain = self._build_chain()

    def _load_prompt(self, promt_path: str) -> Dict[str, Any]:
        with open(promt_path, 'r', encoding='utf-8') as f:
            promt_config = yaml.safe_load(f)
        return promt_config

    def _build_chain(self):
        prompt_template = PromptTemplate(
            template=self.prompt['template'],
            input_variables=self.prompt['input_variables'],
            partial_variables={
                'format_instructions': self.parser.get_format_instructions()
            }
        )
        return prompt_template | self.llm | self.parser

    @staticmethod
    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()

    def node(self, state: CreatorState) -> dict:
        try:
            if state.summary:
                response = self.chain.invoke({'question': state.summary})
            else:
                response = self.chain.invoke({'question': state.chunk})

            content = (
                'Локации: ' + ', '.join(c for c in response.locations))
            tool_msg = ToolMessage(
                content=content,
                tool_call_id=LocationsNode.camel_to_snake(type(self).__name__),
                artifact=response
            )
            msg = state.messages + [tool_msg]

            return {
                'locations': response.locations,
                'messages': msg,
            }
        except Exception as e:
            print(e)
            return {'error': e}
