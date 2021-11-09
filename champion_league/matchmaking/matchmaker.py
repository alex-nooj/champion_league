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
        self_play_prob: float
            The desired probability of choosing the training agent as the opponent.
        league_play_prob: float
            The desired probability of choosing a league agent as the opponent.
        logdir: str
            The path to all of the agents.
        tag: str
            The name of the training agent.
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
        """Function for choosing the opponent.

        Parameters
        ----------
        agent_skill: trueskill.Rating
            The trueskill rating of the training agent.
        trueskills: Dict[str, trueskill.Rating]
            The trueskill ratings of all of the league agents.

        Returns
        -------
        str
            The path to the opponent, or 'self' for self-play.
        """
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
            return self._choose_self()
        elif game_mode == "league":
            return self._choose_league_agent(agent_skill, trueskills)
        elif game_mode == "exploiters":
            return self._choose_exploiter()

    def _choose_self(self) -> str:
        """Function for choosing a version of the training agent for self-play.

        Returns
        -------
        str
            Either the path to a previous version of the agent, or 'self', to use just the current
            agent.
        """
        if np.random.randint(low=0, high=100) < 99:
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
            try:
                opponent = np.random.choice(agents)
                return os.path.join(self.logdir, "challengers", self.tag, opponent)
            except ValueError:
                return "self"

    def _choose_league_agent(
        self, agent_skill: trueskill.Rating, trueskills: Dict[str, trueskill.Rating]
    ) -> str:
        """Function for selecting an agent from the league to serve as the opponent.

        Parameters
        ----------
        agent_skill: trueskill.Rating
            The trueskill rating of the training agent.
        trueskills: Dict[str, trueskill.Rating]
            The trueskill ratings of all of the league agents.

        Returns
        -------
        str
            The path to a league agent.
        """
        agents = os.listdir(os.path.join(self.logdir, "league"))

        if any(agent not in trueskills for agent in agents):
            unplayed_agents = [agent for agent in agents if agent not in trueskills]
            opponent = np.random.choice(unplayed_agents)
        else:
            if np.random.randint(low=0, high=100) < 90:
                # The majority of the time, we want to choose an opponent on the same skill level as
                # the training agent.
                valid_agents = [
                    k
                    for k, v in trueskills.items()
                    if trueskill.quality_1vs1(agent_skill, v) >= 0.50
                ]
            else:
                # At other times, we would like an agent that ours is likely to either destroy or be
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

    def _choose_exploiter(self) -> str:
        """Function for selecting an exploiter for the training agent. Not Implemented.

        Raises
        ------
        NotImplementedError

        Returns
        -------
        str
        """
        raise NotImplementedError
