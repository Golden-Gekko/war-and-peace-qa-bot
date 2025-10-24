from typing import Literal
from pydantic import BaseModel, Field


class Characters(BaseModel):
    characters: Literal[
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
    ] = Field(description='Основные герои')
