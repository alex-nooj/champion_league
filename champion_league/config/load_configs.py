import inspect
from pathlib import Path
from typing import Any
from typing import Dict

from omegaconf import DictConfig
from omegaconf import MissingMandatoryValue
from omegaconf import OmegaConf

from champion_league.network import NETWORKS
from champion_league.preprocessors import PREPROCESSORS
from champion_league.utils.directory_utils import get_save_dir


def get_default_args(filename: str) -> DictConfig:
    path_to_network = Path(filename)
    parent_dir = path_to_network.parent
    config_name = path_to_network.stem + ".yaml"
    path_to_args = parent_dir / config_name
    return OmegaConf.load(path_to_args)


def handle_special_args(args) -> DictConfig:
    args = OmegaConf.merge(args, OmegaConf.from_cli())
    if "config" in args:
        try:
            config_path = Path(args["config"])
        except MissingMandatoryValue:
            pass
        else:
            if not config_path.is_file():
                raise RuntimeError(f"Could not load config file! File does not exist.")
            args = OmegaConf.merge(args, OmegaConf.load(config_path))

    if "network" in args:
        default_network_args = get_default_args(
            inspect.getfile(NETWORKS[args["network"]])
        )
        if args["network"] in args:
            args[args["network"]] = OmegaConf.merge(
                default_network_args, args[args["network"]]
            )
        else:
            args[args["network"]] = default_network_args

    if "preprocessor" in args:
        default_preproc_args = get_default_args(
            inspect.getfile(PREPROCESSORS[args["preprocessor"]])
        )
        if args["preprocessor"] in args:
            args[args["preprocessor"]] = OmegaConf.merge(
                default_preproc_args, args[args["preprocessor"]]
            )
        else:
            args[args["preprocessor"]] = default_preproc_args
    return args


def parse_args(filename: str) -> Dict[str, Any]:
    args = get_default_args(filename)
    args = handle_special_args(args)
    return OmegaConf.to_container(args)


def save_args(agent_dir: Path, epoch: int, args: Dict[str, Any]):
    save_file = get_save_dir(agent_dir, epoch) / "args.yaml"
    OmegaConf.save(config=args, f=str(save_file))
