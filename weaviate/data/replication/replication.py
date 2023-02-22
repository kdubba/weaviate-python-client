from enum import Enum, EnumMeta, auto
from typing import Union


# MetaEnum and BaseEnum are required to support `in` statements:
#    'ALL' in ConsistencyLevel == True
#    12345 in ConsistencyLevel == False
class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            # when item is type ConsistencyLevel
            return item.name in cls.__members__.keys()
        except AttributeError:
            # when item is type str
            return item in cls.__members__.keys()


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class ConsistencyLevel(str, BaseEnum):
    ALL = auto()
    ONE = auto()
    QUORUM = auto()


def name_consistency_level(level: Union[str, ConsistencyLevel]) -> str:
    """
    Returns the name of giving consistency level

    Parameters
    ----------
    level : str or ConsistencyLevel
    Consistency level

    Returns
    -------
    str
        The name of the giving consistency level

    Raises
    ------
    ValueError
        If level is not among valid levels
    """
    if level not in ConsistencyLevel:
        raise ValueError(f"invalid ConsistencyLevel: {level}")
    if isinstance(level, ConsistencyLevel):
        return level.name
    else:
        return level
