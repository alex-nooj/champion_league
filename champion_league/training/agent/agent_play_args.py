from typing import Any
from typing import Dict
from typing import List
from typing import Union

from champion_league.config.load_configs import Args


class AgentPlayArgs(Args):
    def __init__(
        self,
        agent: str,
        battle_format: str,
        nb_actions: int,
        device: int,
        logdir: str,
        tag: str,
        network: str,
        preprocessor: Dict[str, Dict[str, Union[List[str], Dict[str, Dict[str, Any]]]]],
        nb_steps: int,
        epoch_len: int,
        sample_moves: bool,
        opponents: List[str],
        rewards: Dict[str, float],
        agent_args: Dict[str, Any],
        network_args: Dict[str, Any],
    ):
        self.agent = agent
        self.battle_format = battle_format
        self.nb_actions = nb_actions
        self.device = device
        self.logdir = logdir
        self.tag = tag
        self.network = network
        self.preprocessor = preprocessor
        self.nb_steps = nb_steps
        self.epoch_len = epoch_len
        self.sample_moves = sample_moves
        self.opponents = opponents
        self.rewards = rewards
        self.agent_args = agent_args
        self.network_args = network_args

    @property
    def dict_args(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "battle_format": self.battle_format,
            "nb_actions": self.nb_actions,
            "device": self.device,
            "logdir": self.logdir,
            "tag": self.tag,
            "network": self.network,
            "preprocessor": self.preprocessor,
            "nb_steps": self.nb_steps,
            "epoch_len": self.epoch_len,
            "sample_moves": self.sample_moves,
            "opponents": self.opponents,
            "rewards": self.rewards,
            self.network: self.network_args,
            self.agent: self.agent_args,
        }
