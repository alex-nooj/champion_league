from typing import Tuple

import torch
from poke_env.environment.battle import Battle


class Preprocessor:
    def embed_battle(self, battle: Battle) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
