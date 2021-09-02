import json
import os
from typing import Optional, Union, List

import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.agent.scripted.base_scripted import BaseScripted
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.directory_utils import DotDict


class OpponentPlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        opponent: Union[RLOpponent, BaseScripted],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opponent = opponent

    @classmethod
    def from_path(
        cls, path: str, device: Optional[int] = None, **kwargs
    ) -> "OpponentPlayer":
        if device is None:
            device = 0

        with open(os.path.join(path, "args.json"), "r") as fp:
            args = DotDict(json.load(fp))

        if "scripted" in args:
            opponent = SCRIPTED_AGENTS[args.agent]
        else:
            args.resume = False
            network = build_network_from_args(args)
            network.load_state_dict(
                torch.load(
                    os.path.join(path, "network.pt"),
                    map_location=lambda storage, loc: storage,
                )
            )

            preprocessor = build_preprocessor_from_args(args)

            opponent = RLOpponent(
                network=network,
                preprocessor=preprocessor,
                device=f"cuda:{device}",
                sample_moves=False,
            )

        return cls(opponent, **kwargs)

    def choose_move(self, battle: Battle) -> BattleOrder:
        return self.opponent.choose_move(battle)

    @property
    def battle_history(self) -> List[bool]:
        return [b.won for b in self._battles.values()]
