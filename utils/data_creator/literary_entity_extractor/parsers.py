import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException


class ThinkAwarePydanticOutputParser(PydanticOutputParser):
    """PydanticOutputParser с удалением содержимого <think> тегов Qwen3"""
    def parse_result(self, result, *, partial=False):
        result[0].text = self._remove_think_tags(result[0].text)
        try:
            return super().parse_result(result, partial=partial)
        except OutputParserException:
            # print(f'{"*"*50}\nresult - {result[0]}')
            raise

    def parse(self, text: str):
        text_clean = self._remove_think_tags(text)
        return super().parse(text_clean)

    def _remove_think_tags(self, text: str) -> str:
        text_clean = re.sub(r'<think>.*?<\/think>', '', text, flags=re.DOTALL)
        text_clean = re.sub(r'<think>', '', text_clean)
        text_clean = re.sub(r'<\/think>', '', text_clean)
        return text_clean.strip()
