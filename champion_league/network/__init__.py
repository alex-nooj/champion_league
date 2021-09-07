import os

import torch
from torch import nn

from champion_league.network.ability_network import AbilityNetwork
from champion_league.network.gated_encoder import GatedEncoder
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir


networks = {
    "GatedEncoder": GatedEncoder,
    "AbilityNetwork": AbilityNetwork,
}


def build_network_from_args(args: DotDict) -> nn.Module:
    network = networks[args.network].from_args(args).to(f"cuda:{args.device}")

    if args.resume:
        network = resume_network_from_args(args, network)
    return network


def resume_network_from_args(args: DotDict, network: nn.Module) -> nn.Module:
    agent_dir = os.path.join(args.logdir, "challengers", args.tag)
    try:
        network.load_state_dict(
            torch.load(
                os.path.join(
                    get_save_dir(
                        logdir=os.path.join(args.logdir, "challengers"),
                        tag=args.tag,
                        epoch=get_most_recent_epoch(agent_dir),
                    ),
                    "network.pt",
                ),
                map_location=lambda storage, loc: storage,
            )
        )
    except ValueError:
        if os.path.isdir(os.path.join(agent_dir, "sl")):
            network.load_state_dict(
                torch.load(
                    os.path.join(
                        args.logdir,
                        "challengers",
                        args.tag,
                        "sl",
                        "best_model.pt",
                    ),
                    map_location=lambda storage, loc: storage,
                )
            )

    return network
