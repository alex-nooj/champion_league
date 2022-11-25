import typing
from pathlib import Path

import torch
from torch import nn

from champion_league.config import parse_args
from champion_league.preprocessor import Preprocessor
from champion_league.training import league_play
from champion_league.training.agent.agent_play import agent_play
from champion_league.training.agent.agent_play_args import AgentPlayArgs
from champion_league.training.league.league_args import LeagueArgs
from champion_league.utils.build_network_and_preproc import build_network_and_preproc
from champion_league.utils.directory_utils import PokePath


def resume_network(
    agent_path: Path, device: int
) -> typing.Tuple[nn.Module, Preprocessor]:
    """Resumes a network from its last checkpoint.

    Args:
        agent_path: Path to the agent's directory.
        device: The GPU to load the weights onto.

    Returns:
        typing.Tuple[nn.Module, Preprocessor]: Network with the reloaded weights and its
        preprocessor
    """
    epochs = sorted(
        [e for e in agent_path.iterdir() if e.stem.rsplit("_")[-1].isnumeric()]
    )
    network_file = epochs[-1] / "network.pth"
    file_dict = torch.load(network_file, map_loaction=f"cuda:{device}")
    return file_dict["network"], file_dict["preprocessor"]


def train_agent(args: typing.Dict[str, typing.Any]):
    league_path = PokePath(args["logdir"], args["tag"])

    if args["resume"]:
        network, preprocessor = resume_network(league_path.agent, args["device"])
    else:
        network, preprocessor = build_network_and_preproc(args)

    epoch = 0
    if args["mode"]["agent"]:
        agent_play_args = AgentPlayArgs(
            agent=args["agent"],
            battle_format=args["battle_format"],
            nb_actions=args["nb_actions"],
            device=args["device"],
            logdir=args["logdir"],
            tag=args["tag"],
            network=args["network"],
            preprocessor=args["preprocessor"],
            nb_steps=args["agent_play"]["nb_steps"],
            epoch_len=args["agent_play"]["epoch_len"],
            sample_moves=args["agent_play"]["sample_moves"],
            opponents=args["agent_play"]["opponents"],
            rewards=args["rewards"],
            agent_args=args[args["agent"]],
            network_args=args[args["network"]],
        )

        epoch = (
            agent_play(
                preprocessor=preprocessor,
                network=network,
                league_path=league_path,
                args=agent_play_args,
                epoch=epoch,
            )
            + 1
        )

    if args["mode"]["league"]:
        league_args = LeagueArgs(
            agent=args["agent"],
            battle_format=args["battle_format"],
            nb_actions=args["nb_actions"],
            device=args["device"],
            logdir=args["logdir"],
            tag=args["tag"],
            network=args["network"],
            preprocessor=args["preprocessor"],
            resume=args["resume"],
            nb_steps=args["league_play"]["nb_steps"],
            epoch_len=args["league_play"]["epoch_len"],
            entropy_weight=args["league_play"]["entropy_weight"],
            sample_moves=args["league_play"]["sample_moves"],
            rewards=args["rewards"],
            probs=args["league_play"]["probs"],
            agent_args=args[args["agent"]],
            network_args=args[args["network"]],
        )
        league_play(
            preprocessor=preprocessor,
            network=network,
            league_path=league_path,
            args=league_args,
            epoch=epoch,
        )


if __name__ == "__main__":
    import datetime
    import time

    start_time = time.time()
    train_agent(parse_args(__file__))
    print(str(datetime.timedelta(seconds=int(time.time() - start_time))))
