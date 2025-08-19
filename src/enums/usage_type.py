from enum import Enum


class UsageTypeEnum(str, Enum):
    GUEST = "guest"
    NORMAL = "normal"
    PRO = "pro"