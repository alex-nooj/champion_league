from typing import Tuple

from poke_env.environment.battle import Battle
from torch import Tensor


class Op:
    def __init__(self, in_shape: Tuple[int, ...]):
        self._in_shape = in_shape

    def preprocess(self, battle: Battle, state: Tensor) -> Tensor:
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
