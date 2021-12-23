from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle


class Preprocessor:
    def embed_battle(
        self, battle: Battle, reset: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def output_shape(self) -> Dict[str, Tuple[int, ...]]:
        raise NotImplementedError

    def reset(self) -> None:
        pass
