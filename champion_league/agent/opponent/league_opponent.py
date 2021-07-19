import random

import torch
from typing import Optional

from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from champion_league.preprocessors import Preprocessor


class LeagueOpponent:
    def __init__(
        self,
        network: torch.nn.Module,
        preprocessor: Preprocessor,
        sample_moves: Optional[bool] = True,
    ):
        self.network = network
        self.preprocessor = preprocessor
        self.sample_moves = sample_moves

    def choose_move(self, battle: Battle) -> BattleOrder:
        state = self.preprocessor.embed_battle(battle)
        if len(state.size()) == 2:
            state = state.unsqueeze(0)

        y = self.network(state)

        if self.sample_moves:
            action = torch.multinomial(y["action"], 1)
        else:
            action = torch.argmax(y["action"], dim=-1)

        return self._action_to_move(action.item(), battle)

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        if action < 4 and action < len(battle.available_moves) and not battle.force_switch:
            return BattleOrder(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return BattleOrder(battle.available_switches[action - 4])
        else:

            return self.choose_random_move(battle)

    def choose_random_move(self, battle: Battle) -> BattleOrder:
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend([BattleOrder(switch) for switch in battle.available_switches])

        if battle.can_mega_evolve:
            available_orders.extend(
                [BattleOrder(move, mega=True) for move in battle.available_moves]
            )

        if battle.can_dynamax:
            available_orders.extend(
                [BattleOrder(move, dynamax=True) for move in battle.available_moves]
            )

        if battle.can_z_move and battle.active_pokemon:
            available_z_moves = set(battle.active_pokemon.available_z_moves)  # pyre-ignore
            available_orders.extend(
                [
                    BattleOrder(move, z_move=True)
                    for move in battle.available_moves
                    if move in available_z_moves
                ]
            )

        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        else:
            return DefaultBattleOrder()
