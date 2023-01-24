import collections
import typing

import torch
from stable_baselines3.common.callbacks import BaseCallback

from champion_league.utils.directory_utils import PokePath


class LeagueCheckCallback(BaseCallback):
    def __init__(
        self,
        league_path: PokePath,
        epoch_len: int,
        window_size: typing.Optional[int] = 100,
        verbose: typing.Optional[int] = 0,
    ):
        super(LeagueCheckCallback, self).__init__(verbose)
        self._league_path = league_path
        self._league_agents = {
            agent.stem: collections.deque(maxlen=window_size)
            for agent in self._league_path.league.iterdir()
        }
        self._epoch_len = epoch_len
        self.n_finished_battles = 0
        self.n_won_battles = 0
        self.opponent_name = None

    def _on_step(self) -> bool:
        if self.n_finished_battles == self.training_env.envs[0].env.n_won_battles:
            self.opponent_name = self.training_env.envs[0].env.get_opponent().name
        else:
            self.n_finished_battles = self.training_env.envs[0].env.n_finished_battles
            if self.opponent_name is not "self":
                self._league_agents[self.opponent_name].append(
                    int(
                        self.n_won_battles
                        != self.training_env.envs[0].env.n_won_battles
                    )
                )
            self.n_won_battles = self.training_env.envs[0].env.n_won_battles

        if self.n_calls % self._epoch_len == 0:
            agent_win_rates = {
                k: sum(v) / v.maxlen for k, v in self._league_agents.items()
            }
            beating_league = sum([v > 0.5 for v in agent_win_rates.values()]) > 0.7
            if beating_league:
                save_dir = self._league_path.league / self._league_path.agent.stem
                self.model.save(save_dir / "network.zip")
                torch.save(
                    {
                        "preprocessor": self.training_env.envs[0].env.preprocessor,
                        "team": self.training_env.envs[0].env.agent._team,
                    },
                    self._league_path.league
                    / self._league_path.agent.stem
                    / "metadata.zip",
                )
                return False
        return True
