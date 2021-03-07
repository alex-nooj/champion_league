import importlib
import os

import torch
from poke_env.environment.abstract_battle import AbstractBattle

from champion_league.network.linear_three_layer import LinearThreeLayer


class EvalAgent:
    _ACTION_SPACE = None
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001

    def __init__(self, args, tag, path):
        self.tag = tag
        self._device = args.device
        # network_cls = importlib.import_module(args.network)
        self.network = LinearThreeLayer(args.nb_actions).to(self._device)
        checkpoint = torch.load(
            os.path.join(
                path,
                f"{tag}.pt"
            ),
            map_location=lambda storage, loc: storage
        )
        self.network.load_state_dict(checkpoint)
        self.network.eval()

    @torch.no_grad()
    def choose_move(self, battle: AbstractBattle):
        state = self.network.embed_battle(battle).to(self._device).float()
        action = torch.argmax(self.network(state), 0)
        if action.ndim == 0:
            return action.item()
        else:
            return action[0].item()
