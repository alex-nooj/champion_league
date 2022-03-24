from omegaconf import DictConfig
from omegaconf import ListConfig

from champion_league.config import CFG
from champion_league.utils.directory_utils import DotDict


def parse_args() -> DotDict:
    args = {}
    for k, v in CFG.items():
        if isinstance(v, DictConfig):
            args[k] = dict(v)
        elif isinstance(v, ListConfig):
            args[k] = list(v)
        else:
            args[k] = v
    return DotDict(args)
