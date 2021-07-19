import json
import os
from typing import Optional

import torch
from adept.utils.util import DotDict
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from champion_league.agent.opponent.league_opponent import LeagueOpponent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.network import build_network_from_args
from champion_league.preprocessors import Preprocessor, build_preprocessor_from_args


class LeaguePlayer(Player):
    BATTLES = {}

    def __init__(
        self, sample_moves: Optional[bool] = True, **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_moves = sample_moves

        self.opponent = None
        self.mode = None

    def choose_move(self, battle: Battle) -> BattleOrder:
        if self.mode is None:
            raise RuntimeError("Agent cannot be none!")

        return self.opponent.choose_move(battle)

    def change_agent(
        self,
        agent_path: str,
        network: Optional[torch.nn.Module] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> str:
        if network is not None and preprocessor is not None:
            self.opponent = LeagueOpponent(network, preprocessor, self.sample_moves)
            self.mode = "self"
            return "self"

        with open(os.path.join(agent_path, "args.json"), "r") as fp:
            args = json.load(fp)
            args = DotDict(args)
        if "scripted" in args:
            self.mode = "scripted"
            self.opponent = SCRIPTED_AGENTS[args.agent]
        else:
            self.mode = "ml"
            network = build_network_from_args(args).eval()
            network.load_state_dict(torch.load(os.path.join(agent_path, "network.pt")))
            preprocessor = build_preprocessor_from_args(args)
            self.opponent = LeagueOpponent(network, preprocessor, self.sample_moves)

        return args.tag
