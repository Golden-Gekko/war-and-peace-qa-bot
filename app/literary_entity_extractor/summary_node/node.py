import re
import os
from typing import Any, Dict
import yaml

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ..states import CreatorState


class SummaryNode():
    def __init__(
        self,
        llm,
    ):
        self.llm = llm
        self.parser = StrOutputParser()
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
        )
        return prompt_template | self.llm | self.parser

    @staticmethod
    def camel_to_snake(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()

    def node(self, state: CreatorState) -> dict:
        try:
            response = self.chain.invoke({'question': state.chunk})

            content = response
            tool_msg = ToolMessage(
                content=content,
                tool_call_id=SummaryNode.camel_to_snake(type(self).__name__),
            )
            msg = state.messages + [tool_msg]

            return {
                'summary': response,
                'messages': msg,
            }
        except Exception as e:
            return {'error': e}
