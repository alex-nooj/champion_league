import random
from typing import Optional

import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder
from torch import nn

from champion_league.preprocessors import Preprocessor


class RLOpponent:
    def __init__(
        self,
        network: nn.Module,
        preprocessor: Preprocessor,
        device: int,
        sample_moves: Optional[bool] = True,
    ):
        """The league opponent handles all of the move selection logic for the league and none of
        the server connection logic. This allows us to switch opponents at any time without ever
        restarting our server connection.

        Parameters
        ----------
        network: nn.Module
            The network to be used for move selection.
        preprocessor: Preprocessor
            The preprocessor that the network uses.
        device: str
            The device to load the network onto.
        sample_moves: Optional[bool]
            Whether to sample the network's distribution when choosing a move.
        """
        self.network = network.eval()
        self.preprocessor = preprocessor
        self.sample_moves = sample_moves
        self.device = device
        self._prev_internals = {}

    def choose_move(self, battle: Battle) -> BattleOrder:
        """The function used to pass the current state into the network and receive a battle order.

        Parameters
        ----------
        battle: Battle
            The current state in its raw form.

        Returns
        -------
        BattleOrder
            The move that the agent would like to select, converted into a form that is readable by
            PokeEnv and Showdown.
        """
        state = self.preprocessor.embed_battle(battle, False)

        if battle.battle_tag not in self._prev_internals:
            self._prev_internals[battle.battle_tag] = self.network.reset(self.device)

        y, self._prev_internals[battle.battle_tag] = self.network(
            x_internals={
                "x": state,
                "internals": self._prev_internals[battle.battle_tag],
            }
        )

        if self.sample_moves:
            action = torch.multinomial(y["action"][0:], 1)
        else:
            action = torch.argmax(y["action"][0:], dim=-1)

        return self._action_to_move(action.item(), battle)

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        """This function uses the current battle to turn an action into a valid BattleOrder that is
        useable by PokeEnv and Showdown.

        Parameters
        ----------
        action: int
            The action chosen by the agent.
        battle: Battle
            The raw state of the Pokemon battle.

        Returns
        -------
        BattleOrder
            The selected action that is readable by the environment
        """

        # We only allow the agent to use one of its four moves or switch pokemon -- it is not
        # allowed to mega-evolve or dynamax, per Smogon rules for OU
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return BattleOrder(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return BattleOrder(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    @staticmethod
    def choose_random_move(battle: Battle) -> BattleOrder:
        """This allows the agent to choose a random move when the order it would like is unavailable

        Parameters
        ----------
        battle: Battle
            The current, raw state of the Pokemon battle.

        Returns
        -------
        BattleOrder
            The selected action that is readable by the environment.
        """
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        else:
            return DefaultBattleOrder()

    def reset(self):
        self._prev_internals = {}
