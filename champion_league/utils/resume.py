import json
import os
from typing import Optional
from typing import Tuple
import torch
from torch import nn

from adept.utils.util import DotDict

from champion_league.agent.ppo import PPOAgent
from champion_league.network import build_network_from_args


def resume(args: DotDict) -> Tuple[DotDict, nn.Module, PPOAgent]:
    agent_dir = os.path.join(args.logdir, "challengers", args.tag)
    if args.epoch is None:
        args.epoch = max([
            int(e.rsplit("_")[-1])
            for e in os.listdir(agent_dir)
            if e != "sl" and os.path.isdir(os.path.join(agent_dir, e))
        ])

    old_args = reload_old_args(args.logdir, "challengers", args.tag, args.epoch)

    for arg in old_args:
        if arg not in args:
            args[arg] = old_args[arg]

    network = rebuild_network_from_args(
        args, "challengers", args.epoch
    )

    agent = PPOAgent(
        device=args.device,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=os.path.join(args.logdir, "challengers"),
        tag=args.tag,
    )

    agent.save_args(args)
    agent.save_model(agent.network, args.epoch, args)

    agent.reload_tboard(args.epoch)

    return args, network, agent


def reload_old_args(logdir: str, agent_type: str, tag: str, epoch: Optional[int]) -> DotDict:
    """Reloads the arguments from a previous model

    Parameters
    ----------
    logdir: str
        The path to the league directory (i.e., [PATH]/pokemon_trainers/)
    agent_type: str
        Either challengers, exploiters, or league
    tag: str
        The tag of the model
    epoch: Optional[int]
        Which epoch to take the model from

    Returns
    -------
    DotDict
        The original args from the model
    """

    agentdir = os.path.join(logdir, agent_type, tag)

    if epoch is not None and agent_type not in ["challengers", "exploiters"]:
        args_file = os.path.join(agentdir, f"{tag}_{epoch:05d}", "args.json")
    else:
        args_file = os.path.join(agentdir, "args.json")

    with open(args_file, "r") as fp:
        args = DotDict(json.load(fp))

    return args


def rebuild_network_from_args(
    args: DotDict, agent_type: Optional[str] = None, epoch: Optional[int] = None
) -> nn.Module:
    network = build_network_from_args(args)

    agent_dir = os.path.join(
        args.logdir, "challengers" if agent_type is None else agent_type, args.tag
    )

    if epoch is None:
        epochs = [
            int(e.rsplit("_")[-1])
            for e in os.listdir(agent_dir)
            if os.path.isdir(os.path.join(agent_dir, e)) and e != "sl"
        ]
        epoch = max(epochs)

    network.load_state_dict(
        torch.load(
            os.path.join(
                args.logdir,
                "challengers" if agent_type is None else agent_type,
                args.tag,
                f"{args.tag}_{epoch:05d}",
                "network.pt",
            ),
            map_location=lambda storage, loc: storage,
        )
    )

    return network
