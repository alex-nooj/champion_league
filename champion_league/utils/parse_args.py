from champion_league.utils.directory_utils import DotDict
from champion_league.config import CFG


def parse_args() -> DotDict:
    return DotDict({k: v if k != "device" else int(v) for k, v in CFG.items()})
