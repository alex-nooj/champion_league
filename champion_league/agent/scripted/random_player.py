import gc
import json
from typing import List

import orjson
from adept.utils.util import DotDict
from poke_env.player.player import Player


class RandomPlayer(Player):
    BATTLES = {}

    def __init__(self, args: DotDict):
        super().__init__()

    def choose_move(self, battle):
        """Chooses a move for the agent.

        Parameters
        ----------
        battle: Battle
            The raw battle output from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        return self.choose_random_move(battle)
