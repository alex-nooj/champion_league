import collections
import pathlib
import typing

import torch
from poke_env.player import MaxBasePowerPlayer
from poke_env.player import Player
from poke_env.player import RandomPlayer
from poke_env.player import SimpleHeuristicsPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from champion_league.env.opponent_player import OpponentPlayer


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        opponents: typing.List[typing.Union[OpponentPlayer, Player, str, pathlib.Path]],
        verbose: typing.Optional[int] = 0,
    ):
        super(CurriculumCallback, self).__init__(verbose)
        self._opponents = collections.deque(opponents)
        self.n_finished_battles = 0
        self.n_won_battles = 0
        self._wins = collections.deque(maxlen=100)

    def _on_training_start(self) -> None:
        self._set_next_opponent()

    def _on_step(self) -> bool:
        if self.n_finished_battles != self.training_env.envs[0].n_finished_battles:
            self._wins.append(
                int(self.n_won_battles != self.training_env.envs[0].env.n_won_battles)
            )
            self.n_won_battles = self.training_env.envs[0].env.n_won_battles
            self.n_finished_battles = self.training_env.envs[0].n_finished_battles
            self._logging()
            if sum(self._wins) / self._wins.maxlen >= 0.6 and len(self._opponents) > 0:
                self._set_next_opponent()
                self._wins = collections.deque(maxlen=100)
            elif (
                sum(self._wins) / self._wins.maxlen >= 0.6 and len(self._opponents) == 0
            ):
                return False
        return True

    def _set_next_opponent(self):
        opponent = self._opponents.popleft()
        if isinstance(opponent, str):
            opponent = pathlib.Path(opponent)

        if isinstance(opponent, pathlib.Path):
            opponent_model = PPO.load(opponent / "network.zip")
            metadata = torch.load(opponent / "metadata.zip")
            opponent = OpponentPlayer(
                preprocessor=metadata["preprocessor"],
                model=opponent_model,
                battle_format=self.training_env.envs[0].env.format,
                team=metadata["team"],
                name=opponent.stem,
            )
        self.training_env.envs[0].env.set_opponent(opponent)

    def _logging(self):
        opponent = self.training_env.envs[0].env.get_opponent()
        if isinstance(opponent, RandomPlayer):
            opponent_name = "RandomPlayer"
        elif isinstance(opponent, MaxBasePowerPlayer):
            opponent_name = "MaxBasePowerPlayer"
        elif isinstance(opponent, SimpleHeuristicsPlayer):
            opponent_name = "SimpleHeuristicsPlayer"
        else:
            opponent_name = opponent.name

        if self.verbose == 1:
            print(f"{opponent_name}: {sum(self._wins) / self._wins.maxlen: 0.3f}")
        elif self.verbose == 2:
            self.logger.record(
                f"Curriculum/{opponent_name}",
                sum(self._wins) / self._wins.maxlen,
            )
