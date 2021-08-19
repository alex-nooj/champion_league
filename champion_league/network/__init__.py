from adept.utils.util import DotDict
from torch import nn


def build_network_from_args(args: DotDict) -> nn.Module:
    import importlib

    network_path = f"champion_league.network.{args.network}"
    build_cls = getattr(importlib.import_module(network_path), "build_from_args")
    return build_cls(args).to(f"cuda:{args.device}")
