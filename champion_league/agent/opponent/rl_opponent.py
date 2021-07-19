import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from typing import Optional

from champion_league.agent.scripted import SimpleHeuristic
from champion_league.preprocessors import Preprocessor


class RLOpponent(Player):
    BATTLES = {}

    def __init__(
        self,
        network: torch.nn.Module,
        preprocessor: Preprocessor,
        sample_moves: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.network = network
        self.preprocessor = preprocessor
        self.sample_moves = sample_moves
        self.eval_agent = SimpleHeuristic("simple_heuristic")
        self._eval = False

    def choose_move(self, battle: Battle) -> BattleOrder:
        if self._eval:
            return self.eval_agent.choose_move(battle)
        else:
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
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    def eval(self):
        self._eval = True

    def train(self):
        self._eval = False
