import pathlib
import typing

import numpy as np
import torch
import trueskill
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from champion_league.env.opponent_player import OpponentPlayer
from champion_league.utils.directory_utils import PokePath


class SkillTracker:
    def __init__(self, league_dir: pathlib.Path):
        self._league_dir = league_dir
        self.agent_skill = trueskill.Rating()
        self.league_skills = {}
        for agent in self._league_dir.iterdir():
            trueskill_file = agent / "trueskill.yaml"
            if trueskill_file.is_file():
                with open(trueskill_file, "r") as fp:
                    self.league_skills[agent.stem] = trueskill.Rating(
                        **yaml.safe_load(fp)
                    )
            else:
                print(f"{agent.stem} has no recorded trueskill. Using default instead.")
                self.league_skills[agent.stem] = trueskill.Rating()

    def update_skills(self, agent_won: bool, opponent: str):
        if agent_won:
            self.agent_skill, self.league_skills[opponent] = trueskill.rate_1vs1(
                self.agent_skill,
                self.league_skills[opponent],
            )

    def save_skills(self):
        for agent, skill in self.league_skills.items():
            with open(self._league_dir / agent / "trueskill.yaml", "w") as fp:
                yaml.dump({"mu": skill.mu, "sigma": skill.sigma}, fp)


class MatchMakingCallback(BaseCallback):
    def __init__(
        self,
        league_path: PokePath,
        self_play_prob: float,
        league_play_prob: float,
        verbose: typing.Optional[int] = 0,
    ):
        super(MatchMakingCallback, self).__init__(verbose)
        self.game_mode_probs = {
            "challengers": self_play_prob,
            "league": league_play_prob,
            "exploiters": 1 - self_play_prob - league_play_prob,
        }
        self._league_path = league_path

        self.n_finished_battles = 0
        self.n_won_battles = 0

        self._current_opponent = None

        self._agent_skill = trueskill.Rating()
        self._league_skills = {}
        for agent in self._league_path.league.iterdir():
            trueskill_file = agent / "trueskill.yaml"
            if trueskill_file.is_file():
                with open(trueskill_file, "r") as fp:
                    self._league_skills[agent.stem] = trueskill.Rating(
                        **yaml.safe_load(fp)
                    )
            else:
                print(f"{agent.stem} has no recorded trueskill. Using default instead.")
                self._league_skills[agent.stem] = trueskill.Rating()

    def _on_training_start(self) -> None:
        """Collect all the agents and choose our first opponent."""
        self.training_env.envs[0].env.set_opponent(
            self._choose_match(self._agent_skill, self._league_skills)
        )

    def _on_step(self) -> bool:
        """Check if the agent won and update the skills. If the battle ended, update to the next
        opponent.

        Returns:
            bool: True to continue training, False to raise an error.
        """
        if self.n_finished_battles != self.training_env.envs[0].n_finished_battles:
            agent_won = (
                self.n_won_battles == self.training_env.envs[0].env.n_won_battles
            )
            if agent_won:
                (
                    self._agent_skill,
                    self._league_skills[self._current_opponent],
                ) = trueskill.rate_1vs1(
                    self._agent_skill,
                    self._league_skills[self._current_opponent],
                )
            else:
                (
                    self._league_skills[self._current_opponent],
                    self._agent_skill,
                ) = trueskill.rate_1vs1(
                    self._league_skills[self._current_opponent],
                    self._agent_skill,
                )
            self.training_env.envs[0].env.set_opponent(
                self._choose_match(self._agent_skill, self._league_skills)
            )
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        """Save all the agent skills."""
        for agent, skill in self._league_skills.items():
            with open(self._league_path.league / agent / "trueskill.yaml", "w") as fp:
                yaml.dump({"mu": skill.mu, "sigma": skill.sigma}, fp)

    def _choose_match(
        self,
        agent_skill: trueskill.Rating,
        league_skills: typing.Dict[str, trueskill.Rating],
    ) -> OpponentPlayer:
        mode_probs = []
        mode_options = []
        for agent_type in [
            self._league_path.challengers,
            self._league_path.league,
            self._league_path.exploiters,
        ]:
            mode_options.append(agent_type.stem)
            mode_probs.append(self.game_mode_probs[agent_type.stem])
        mode_probs = [prob / sum(mode_probs) for prob in mode_probs]
        game_mode = np.random.choice(mode_options, p=mode_probs)
        if game_mode == "challengers":
            return self._choose_self()
        elif game_mode == "league":
            return self._choose_league(agent_skill, league_skills)
        else:
            return self._choose_exploiter()

    def _choose_self(self) -> OpponentPlayer:
        return OpponentPlayer(
            preprocessor=self.training_env.envs[0].env.preprocessor,
            model=self.model,
            battle_format=self.training_env.envs[0].env.format,
            team=self.training_env.envs[0].env.agent._team,
            name="self",
        )

    def _choose_league(
        self,
        agent_skill: trueskill.Rating,
        league_skills: typing.Dict[str, trueskill.Rating],
    ) -> OpponentPlayer:
        if np.random.randint(low=0, high=100) < 90:
            # The majority of the time we want to choose an agent on the same skill level as the
            # training agent
            valid_agents = [
                k
                for k, v in league_skills.items()
                if trueskill.quality_1vs1(agent_skill, v) >= 0.5
            ]
        else:
            valid_agents = [
                k
                for k, v in league_skills.items()
                if trueskill.quality_1vs1(agent_skill, v) < 0.5
            ]
        if len(valid_agents) == 0:
            valid_agents = [k for k in league_skills]

        opponent_path = self._league_path.league / np.random.choice(valid_agents)
        opponent_model = PPO.load(opponent_path / "network.zip")
        metadata = torch.load(opponent_path / "metadata.zip")
        return OpponentPlayer(
            preprocessor=metadata["preprocessor"],
            model=opponent_model,
            battle_format=self.training_env.envs[0].env.format,
            team=metadata["team"],
            name=opponent_path.stem,
        )

    def _choose_exploiter(self):
        raise NotImplementedError()
