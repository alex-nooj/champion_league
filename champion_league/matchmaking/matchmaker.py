import os
from typing import Dict

import numpy as np
import trueskill


class MatchMaker:
    def __init__(
        self,
        self_play_prob: float,
        league_play_prob: float,
        logdir: str,
        tag: str,
    ):
        """This class handles all the logic in selecting an opponent for the league.

        Parameters
        ----------
        self_play_prob
        league_play_prob
        logdir
        tag
        """
        if 1 - self_play_prob - league_play_prob < 0:
            raise RuntimeError("Self Play and League Play Probs are too high!")

        self.game_mode_probs = {
            "challengers": self_play_prob,
            "league": league_play_prob,
            "exploiters": 1 - self_play_prob - league_play_prob,
        }

        self.self_play_prob = self_play_prob
        self.league_play_prob = league_play_prob
        self.logdir = logdir
        self.tag = tag

    def choose_match(
        self, agent_skill: trueskill.Rating, trueskills: Dict[str, trueskill.Rating]
    ) -> str:
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
            if np.random.randint(low=0, high=100) < 90:
                return "self"
            else:
                agents = os.listdir(os.path.join(self.logdir, "challengers", self.tag))
                agents = [
                    epoch
                    for epoch in agents
                    if os.path.isdir(
                        os.path.join(self.logdir, "challengers", self.tag, epoch)
                    )
                    and epoch != "sl"
                ]
                opponent = np.random.choice(agents)
                return os.path.join(self.logdir, "challengers", self.tag, opponent)
        elif game_mode == "league":
            # League play
            agents = os.listdir(os.path.join(self.logdir, "league"))
            if any(agent not in trueskills for agent in agents):
                unplayed_agents = [agent for agent in agents if agent not in trueskills]
                opponent = np.random.choice(unplayed_agents)
            else:
                if np.random.randint(low=0, high=100) < 90:
                    # The majority of the time, we want to choose an opponent that our agent is at a
                    # similar skill to our agent
                    valid_agents = [
                        k
                        for k, v in trueskills.items()
                        if trueskill.quality_1vs1(agent_skill, v) >= 0.50
                    ]
                else:
                    # Other times, we want an agent that ours is likely to either destroy, or get
                    # destroyed by
                    valid_agents = [
                        k
                        for k, v in trueskills.items()
                        if trueskill.quality_1vs1(agent_skill, v) < 0.50
                    ]
                if len(valid_agents) == 0:
                    valid_agents = [k for k in trueskills]
                opponent = np.random.choice(valid_agents)
            return os.path.join(self.logdir, "league", opponent)
        elif game_mode == "exploiters":
            raise NotImplementedError
