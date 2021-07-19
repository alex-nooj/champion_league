import os
from typing import Dict, Optional, List

import numpy as np


class MatchMaker:
    def __init__(
        self,
        self_play_prob: float,
        league_play_prob: float,
        logdir: str,
        tag: str,
        alpha: Optional[float] = 50,
    ):
        assert (
            1 - self_play_prob - league_play_prob >= 0,
            "Self Play and League Play Probs are too high!",
        )
        self.game_mode_probs = {
            "challengers": self_play_prob,
            "league": league_play_prob,
            "exploiters": 1 - self_play_prob - league_play_prob,
        }

        self.self_play_prob = self_play_prob
        self.league_play_prob = league_play_prob
        self.logdir = logdir
        self.tag = tag
        self.alpha = alpha

    def choose_match(self, win_rates: Dict[str, List[int]]) -> str:
        mode_probs = []
        mode_options = []
        for agent_type in ["challengers", "league", "exploiters"]:
            if len(os.listdir(os.path.join(self.logdir, agent_type))) > 0:
                mode_options.append(agent_type)
                mode_probs.append(self.game_mode_probs[agent_type])

        # Re-normalize the probabilities
        mode_probs = [prob / np.sum(mode_probs) for prob in mode_probs]

        # Choose the game mode
        game_mode = np.random.choice(mode_options, p=mode_probs)

        if game_mode == "challengers":
            # Self-play
            if np.random.randint(low=0, high=100) > 80:
                agents = os.listdir(os.path.join(self.logdir, "challengers", self.tag))
                agents = [
                    epoch
                    for epoch in agents
                    if os.path.isdir(os.path.join(self.logdir, "challengers", self.tag, epoch))
                ]
                opponent = np.random.choice(agents)
                return os.path.join(self.logdir, "challengers", self.tag, opponent)
            else:
                return "self"
        elif game_mode == "league":
            # League play
            agents = os.listdir(os.path.join(self.logdir, "league"))
            if any(agent not in win_rates for agent in agents):
                unplayed_agents = [agent for agent in agents if agent not in win_rates]
                opponent = np.random.choice(unplayed_agents)
            else:
                league_agents = [key for key in win_rates if key in agents]
                adjusted_rates = [
                    self.alpha ** (-1 * v[0] / v[1]) for k, v in win_rates.items() if k in agents
                ]
                adjusted_rates = [v / np.sum(adjusted_rates) for v in adjusted_rates]
                opponent = np.random.choice(league_agents, p=adjusted_rates)
            return os.path.join(self.logdir, "league", opponent)
        elif game_mode == "exploiters":
            raise NotImplementedError
