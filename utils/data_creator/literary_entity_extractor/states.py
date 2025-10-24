from typing import List

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class CreatorState(BaseModel):
    chunk: str = Field(
        description='Текущий чанк для обработки'
    )
    characters: List[str] = Field(
        default=[],
        description='Список персонажей'
    )
    locations: List[str] = Field(
        default=[],
        description='Список локаций'
    )
    summary: List[str] = Field(
        default=[],
        description='Суммаризированный моделью чанк'
    )
    messages: List[BaseMessage] = Field(
        default=[],
        description='История сообщений'
    )

    @classmethod
    def create(cls, chunk: str = ''):
        return cls(chunk=chunk)
