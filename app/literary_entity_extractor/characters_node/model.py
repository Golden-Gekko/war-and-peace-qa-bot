from typing import List, Literal
from pydantic import BaseModel, Field


class Characters(BaseModel):
    characters: List[
        Literal[
            'Андрей Болконский',
            'Наташа Ростова',
            'Пьер Безухов',
            'Николай Ростов',
            'Илья Ростов',
            'Наталья Ростова',
            'Николай Болконский',
            'Марья Болконская',
            'Федор Долохов',
            'Василий Денисов',
            'Соня'
        ]
    ] = Field(
        default_factory=list,
        description='Список основных героев, упомянутых в тексте')
