from typing import Optional
import torch
from typing import Tuple

from champion_league.preprocessors.modules.basemodule import BaseModule
from champion_league.preprocessors.modules.battle_to_tensor import BattleIdx, NB_POKEMON


class EmbedAbilities(BaseModule):
    def embed(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, :, BattleIdx.ability].squeeze(-1)

    @property
    def output_shape(self) -> Tuple[int]:
        return (NB_POKEMON,)
