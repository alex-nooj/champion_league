import pathlib
import random
import typing

import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.player.player import Player


class OpponentPlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        path: pathlib.Path,
        device: typing.Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if device is None:
            device = "cpu"
        elif isinstance(device, int):
            device = f"cuda:{device}"

        agent_data = torch.load(path / "network.pth", map_location=device)
        self.network = agent_data["network"]
        self.preprocessor = agent_data["preprocessor"]
        self.team = agent_data["team"]

    def choose_move(self, battle: Battle) -> BattleOrder:
        """Function that allows the agent to select a move.

        Args:
            battle: The current game state.

        Returns:
            BattleOrder: The action the agent would like to take, in a format readable by Showdown!
        """
        state = self.preprocessor.embed_battle(battle)

        with torch.no_grad():
            y = self.network(x=state)
        action = torch.argmax(y["action"][0:], dim=-1).item()
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

    def choose_random_move(self, battle: Battle) -> BattleOrder:
        """This allows the agent to choose a random move when the order it would like is unavailable

        Args:
        battle: The current, raw state of the Pokemon battle.

        Returns:
            BattleOrder: The selected action that is readable by the environment.
        """
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        else:
            return DefaultBattleOrder()

    @property
    def battle_history(self) -> typing.List[bool]:
        """Returns a list containing the win/loss results of the agent.

        Returns:
            List[bool]: Contains the win/loss history of the agent.
        """
        return [b.won for b in self._battles.values()]
