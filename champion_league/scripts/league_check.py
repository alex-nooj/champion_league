from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

from champion_league.agent.ppo import PPOAgent
from champion_league.config import parse_args
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.network import NETWORKS
from champion_league.preprocessors import PREPROCESSORS
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.scripts.league_play import move_to_league
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.progress_bar import centered
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


def print_wins(win_rates: Dict[int, Dict[str, int]]):
    """Prints out a pretty table with the win rates against each league agent for each epoch.

    Parameters
    ----------
    win_rates: Dict[int, Dict[str, int]]
        Dict with all of the epochs as keys and a second Dict as values. The second Dict uses the
        league agent names as keys and the number of wins as values.

    Returns
    -------
    None
    """
    agent_names = [name for name in win_rates[0]]
    max_name_length = max([len(name) for name in agent_names])
    header = "|" + centered("Agent", max_name_length + 2) + "|"
    divider = "+" + "-" * len(centered("Agent", max_name_length + 2)) + "+"
    for epoch in win_rates:
        header += centered(str(epoch), 5)
        header += "|"
        divider += "-" * 5
        divider += "+"

    print(divider)
    print(header)
    print(divider)
    for agent in agent_names:
        agent_line = "|" + centered(agent, max_name_length + 2) + "|"
        for epoch in win_rates:
            agent_line += centered(str(win_rates[epoch][agent]), 5)
            agent_line += "|"
        print(agent_line)
    print(divider)


def agent_check(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    logdir: str,
    agent_epochs: List[int],
    step_counter: StepCounter,
    nb_battles: int,
):
    """Main loop for pitting every epoch of the agent against every league agent, then printing the
    results.

    Parameters
    ----------
    player: RLPlayer
        The player that is actually running the game. This acts as our "environment" and has the
        `step()` and `reset()` functions.
    agent: PPOAgent
        The agent that is playing the game. Overkill to use here, but it ensures that internals are
        being handled correctly.
    opponent: LeaguePlayer
        The agent that is playing against the training agent. This agent handles scheme selection,
        but its actual action selection is being done in a separate thread.
    logdir: str
        The path to all of the agents.
    agent_epochs: List[int]
        A list of all the epochs of the agent in question.
    step_counter: StepCounter
        Tracks the total number of steps across each epoch.
    nb_battles: int
        How many battles the agent should play against each league agent.

    Returns
    -------
    None
    """
    win_rates = {}
    league_agents = list(Path(logdir, "league"))
    league_agents.sort()
    for epoch in agent_epochs:
        print(f"Testing Epoch {epoch}...")
        agent.network = load_epoch(logdir, agent.tag, epoch, agent.network)
        win_rates[epoch] = {}
        for league_agent in tqdm(league_agents):
            win_rates[epoch][league_agent] = 0
            _ = opponent.change_agent(league_agent)
            for _ in range(nb_battles):
                episode = collect_episode(
                    player=player, agent=agent, step_counter=step_counter
                )
                win_rates[epoch][league_agent.stem] += int(episode.rewards[-1] > 0)
        agents_beaten = np.sum(
            [
                int(win_rate > (nb_battles * 0.5))
                for win_rate in win_rates[epoch].values()
            ]
        )
        print(f"Epoch {epoch:3d}: {agents_beaten}")
        if agents_beaten >= (len(league_agents) * 0.75):
            move_to_league(
                Path(logdir, "challengers"),
                Path(logdir, "league"),
                agent.tag,
                epoch,
            )

    print_wins(win_rates)


def get_old_args(logdir: str, tag: str) -> Dict[str, Any]:
    """Retrieves the arguments of a previously trained agent.

    Parameters
    ----------
    logdir: str
        The path to all of the agents.
    tag: str
        The tag of the agent you would like to load.

    Returns
    -------
    DotDict
        DotDict containing all of the arguments.
    """

    return OmegaConf.to_container(
        OmegaConf.load(Path(logdir, "challengers", tag, "args.yaml"))
    )


def load_epoch(logdir: str, tag: str, epoch: int, network: nn.Module) -> nn.Module:
    """Loads the network weights from a specific epoch.

    Parameters
    ----------
    logdir: str
        The path to all of the agents.
    tag: str
        The tag of the agent you would like to load.
    epoch: int
        The epoch to load from.
    network: nn.Module
        The network to load the weights into.

    Returns
    -------
    nn.Module
        The network with weights loaded.
    """
    network.load_state_dict(
        torch.load(
            get_save_dir(Path(logdir, tag), epoch) / "network.pt",
        )
    )
    return network


def main(logdir: str, tag: str, nb_battles: int):
    """The main function for checking an agent's performance against all of the league agents.

    Parameters
    ----------
    logdir: str
        The path to all of the agents.
    tag: str
        The tag of the agent you would like to load.
    nb_battles: int
        How many battles the agent should play against each league agent.

    Returns
    -------
    None
    """
    args = get_old_args(logdir, tag)

    agent_dir = Path(logdir, "challengers", args["tag"])

    preprocessor = PREPROCESSORS[args["preprocessor"]](
        args["device"], **args[args["preprocessor"]]
    )

    network = NETWORKS[args["network"]](
        args["nb_actions"], preprocessor.output_shape, **args[args["network"]]
    ).eval()
    reward_scheme = RewardScheme(rules=args["rewards"])

    agent_epochs = [
        int(e.stem.rsplit("_")[-1])
        for e in agent_dir.iterdir()
        if e.is_dir() and e.stem != "sl"
    ]

    step_counter = StepCounter()
    agent_epochs.sort()

    league_agents = [p for p in Path(logdir, "league").iterdir()]
    league_agents.sort()

    agent = PPOAgent(
        device=args["device"],
        network=network,
        lr=args["lr"],
        entropy_weight=args["entropy_weight"],
        clip=args["clip"],
        challenger_dir=Path(args["logdir"], "challengers"),
        tag=args["tag"],
    )

    player = RLPlayer(
        battle_format=args["battle_format"],
        embed_battle=preprocessor.embed_battle,
        reward_scheme=reward_scheme,
        server_configuration=DockerServerConfiguration,
    )

    opponent = LeaguePlayer(
        device=agent.device,
        network=network,
        preprocessor=preprocessor,
        sample_moves=False,
        max_concurrent_battles=10,
        server_configuration=DockerServerConfiguration,
    )

    player.play_against(
        env_algorithm=agent_check,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "opponent": opponent,
            "logdir": logdir,
            "agent_epochs": agent_epochs,
            "step_counter": step_counter,
            "nb_battles": nb_battles,
        },
    )


if __name__ == "__main__":
    main(**parse_args(__file__))
