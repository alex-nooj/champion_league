import typing

import torch
from stable_baselines3.common.callbacks import BaseCallback

from champion_league.utils.directory_utils import PokePath


class SaveCallback(BaseCallback):
    def __init__(
        self, league_path: PokePath, epoch_len: int, verbose: typing.Optional[int] = 0
    ):
        super(SaveCallback, self).__init__(verbose)
        self._league_path = league_path
        self._epoch_len = epoch_len

    def _on_training_start(self) -> None:
        self._league_path.agent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "preprocessor": self.training_env.envs[0].env.preprocessor,
                "team": self.training_env.envs[0].env.agent._team,
            },
            self._league_path.agent / "metadata.zip",
        )

        (self._league_path.agent / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._epoch_len == 0:
            save_dir = (
                self._league_path.agent
                / "checkpoints"
                / f"network_{self.n_calls // self._epoch_len:05d}.zip"
            )
            self.model.save(save_dir)

            if self.verbose == 1:
                print(f"{self.n_calls}: Model saved to {save_dir}")
        return True

    def _on_training_end(self) -> None:
        save_dir = self._league_path.agent / "checkpoints" / "network_final.zip"
        self.model.save(save_dir)

        if self.verbose == 1:
            print(f"{self.n_calls}: Model saved to {save_dir}")
