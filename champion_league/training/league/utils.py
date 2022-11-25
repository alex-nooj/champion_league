import asyncio
from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np
from poke_env.player_configuration import PlayerConfiguration
from tqdm import tqdm

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.env import LeaguePlayer
from champion_league.env import OpponentPlayer
from champion_league.matchmaking.league_skill_tracker import LeagueSkillTracker
from champion_league.preprocessor import Preprocessor
from champion_league.teams.team_builder import AgentTeamBuilder
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.progress_bar import centered


def move_to_league(agent_path: Path, league_dir: Path, tag: str, epoch: int) -> None:
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


def league_score(
    agent: Agent,
    preprocessor: Preprocessor,
    opponent: LeaguePlayer,
    league_dir: Path,
    skill_tracker: LeagueSkillTracker,
    team_builder: AgentTeamBuilder,
    nb_battles: Optional[int] = 100,
) -> float:
    """Function for determining how many of the league agents are considered 'beaten'.

    Args:
        agent: The currently training agent.
        preprocessor: The preprocessing scheme for the current agent.
        opponent: The player for handling all the league logic.
        league_dir: The path to the league agents.
        skill_tracker: Helper class for handling the trueskill of all the agents.
        team_builder: The agent's team builder.
        nb_battles: The number of battles used to determine the agent's win rates. Default: 100

    Returns:
        float: Percentage of the league that the agent has over a 50% win rate against.
    """
    print("starting score")
    challenger = OpponentPlayer(
        opponent=RLOpponent(
            network=agent.network,
            preprocessor=preprocessor,
            device=agent.device,
            sample_moves=False,
        ),
        max_concurrent_battles=100,
        player_configuration=PlayerConfiguration(
            username="rlchallenger", password="rlchallenger1234"
        ),
        team=team_builder,
    )

    print("challenger made")
    sample_moves = opponent.sample_moves
    opponent.sample_moves = False

    overall_wins = 0
    win_dict = {}
    for league_agent in tqdm(league_dir.iterdir()):
        print(league_agent)
        # To eval, we play 100 games against each agent in the league. If the agent wins over 50
        # games against 75% of the opponents, it is added to the league as a league agent.
        a = opponent.change_agent(league_agent)
        print(a)
        asyncio.get_event_loop().run_until_complete(
            league_match(challenger, opponent, nb_battles=nb_battles)
        )

        for result in challenger.battle_history:
            skill_tracker.update(result, league_agent.stem)
        agent.log_scalar(
            f"League Validation/{league_agent.stem}",
            challenger.win_rate * nb_battles,
        )
        win_dict[league_agent.stem] = challenger.win_rate

        overall_wins += int(challenger.win_rate >= 0.5)
        challenger.reset_battles()
    print_table(win_dict)
    opponent.sample_moves = sample_moves
    return overall_wins / len(list(league_dir.iterdir()))


def print_table(entries: Dict[str, float], float_precision: Optional[int] = 1) -> None:
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


async def league_match(
    challenger: OpponentPlayer,
    opponent: LeaguePlayer,
    nb_battles: Optional[int] = 100,
) -> None:
    """Asynchronous function for handling one player battling another.

    Args:
        challenger: The training agent.
        opponent: The player used to load in all the League schemes.
        nb_battles: How many battles to perform.

    """
    await challenger.battle_against(opponent, n_battles=nb_battles)
