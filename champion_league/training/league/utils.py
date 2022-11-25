import pathlib
import typing

import numpy as np

from champion_league.agent.base.base_agent import Agent
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.progress_bar import centered


def move_to_league(
    agent_path: pathlib.Path, league_dir: pathlib.Path, tag: str, epoch: int
) -> None:
    """Adds the files necessary to build an agent into the league directory

    Args:
        agent_path: The path to the agent
        league_dir: The path to the league
        tag: The name of the agent
        epoch: The epoch of training
    """
    try:
        league_agent = get_save_dir(league_dir / tag, epoch, False)
        league_agent = league_dir / league_agent.stem
        league_agent.symlink_to(
            get_save_dir(agent_path, epoch), target_is_directory=True
        )
    except FileExistsError:
        pass


def print_table(
    entries: typing.Dict[str, float], float_precision: typing.Optional[int] = 1
) -> None:
    """Used for printing a table of win rates against agents

    Args:
        entries: Dictionary with agent ID's for keys and win rates for values
        float_precision: How many values to show after the decimal point
    """
    header1 = "League Agent"
    header2 = "Win Rate"

    max_string_length = max([len(header1)] + [len(key) for key in entries]) + 2

    header2_length = max([len(header2), float_precision + 7]) + 2
    divider = "+" + "-" * max_string_length + "+" + "-" * header2_length + "+"

    print(divider)

    print(
        "|"
        + centered(header1, max_string_length)
        + "|"
        + centered(header2, header2_length)
        + "|"
    )

    print(divider)

    for entry, v in entries.items():
        print(
            "|"
            + centered(entry, max_string_length)
            + "|"
            + centered(f"{v*100:0.1f}%", header2_length)
            + "|"
        )

    print(divider)


def beating_league(agent: Agent) -> bool:
    win_rates = {k: np.mean(v) for k, v in agent.win_rates.items() if k != "self"}
    print_table(win_rates)
    return np.mean([int(v >= 0.5) for v in win_rates.values()]) >= 0.75
