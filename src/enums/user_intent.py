from typing import List
from enum import Enum


class UserIntentEnum(str, Enum):
    TELL: str = "TELL"
    CONTINUE: str = "CONTINUE"
    SUMMARY: str = "SUMMARY"
    OTHER: str = "OTHER"

    @classmethod
    def get_list_value(cls) -> List[str]:
        return cls.__dict__['_member_names_']
    

    