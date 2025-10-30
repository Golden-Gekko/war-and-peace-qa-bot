from typing import List, Literal
from pydantic import BaseModel, Field


class Locations(BaseModel):
    locations: List[
        Literal[
            'Москва',
            'Санкт-Петербург',
            'Болдино',
            'Лысые Горы',
            'Отрадное',
            'Аустерлиц',
            'Бородино',
            'Вильна',
            'Смоленск',
            'Тарутино'
        ]
    ] = Field(
        default_factory=list,
        description='Список основных локаций, упомянутых в тексте')
