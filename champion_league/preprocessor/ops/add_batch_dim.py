from typing import Tuple

from poke_env.environment.battle import Battle
from torch import Tensor

from champion_league.preprocessor.ops.base_op import Op


class AddBatchDim(Op):
    def preprocess(self, battle: Battle, state: Tensor) -> Tensor:
        return state.unsqueeze(0)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._in_shape
