import json
import os
from typing import List
from typing import Optional
from typing import Union

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
        """Creates the player from a given path

        Parameters
        ----------
        path: str
            The path to the desired agents directory
        device: Optional[int]
            The GPU to load the agent on. If None, defaults to 0
        kwargs: Dict[Any]
            Keyword arguments to be passed to the Player class

        Returns
        -------
        "OpponentPlayer"
            Player that can connect to the server and play against a human or another agent.
        """
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
        """Function that allows the agent to select a move.

        Parameters
        ----------
        battle: Battle
            The current game state.

        Returns
        -------
        BattleOrder
            The action the agent would like to take, in a format readable by Showdown!
        """
        return self.opponent.choose_move(battle)

    @property
    def battle_history(self) -> List[bool]:
        """Returns a list containing the win/loss results of the agent.

        Returns
        -------
        List[bool]
            Contains the win/loss history of the agent.
        """
        return [b.won for b in self._battles.values()]
