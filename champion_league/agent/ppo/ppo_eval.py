import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player


class PPOEval(Player):
    def __init__(
            self,
            network: torch.nn.Module,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.network = network

    def choose_move(self, battle: Battle) -> BattleOrder:

    def act(self, state: torch.Tensor) -> int:
        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        return torch.argmax(self.network(state)["action"], -1).item()
