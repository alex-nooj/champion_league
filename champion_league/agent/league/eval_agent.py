import importlib
import os
from typing import Dict

import torch
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle

from champion_league.network.linear_three_layer import LinearThreeLayer
from champion_league.network.lstm_network import LSTMNetwork


class EvalAgent:
    _ACTION_SPACE = None
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001

    def __init__(self, args, tag, path):
        self.tag = tag
        self._device = args.device
        # network_cls = importlib.import_module(args.network)
        # self.network = LinearThreeLayer(args.nb_actions).to(self._device)
        self.network = LinearThreeLayer(args.nb_actions).to(self._device)

        checkpoint = torch.load(
            os.path.join(path, f"{tag}.pt"), map_location=lambda storage, loc: storage
        )
        self.network.load_state_dict(checkpoint)
        self.network.eval()

    @torch.no_grad()
    def choose_move(self, battle: Battle, internals: Dict[str, torch.Tensor]):
        state = self.network.embed_battle(battle).to(self._device).float()
        pred, new_internals = self.network(state, internals)
        action = torch.argmax(pred["action"], 0)
        if action.ndim == 0:
            return action.item(), new_internals
        else:
            return action[0].item(), new_internals
