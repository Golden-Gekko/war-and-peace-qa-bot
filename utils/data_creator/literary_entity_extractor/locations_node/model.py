from typing import Literal
from pydantic import BaseModel, Field


class Locations(BaseModel):
    locations: Literal[
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
    ] = Field(description='Основные локации')
