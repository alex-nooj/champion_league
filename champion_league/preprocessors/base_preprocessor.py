import torch
from poke_env.environment.battle import Battle
from typing import Tuple


class Preprocessor:
    def embed_battle(self, battle: Battle) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
