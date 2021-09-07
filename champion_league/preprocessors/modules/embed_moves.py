from enum import auto
from enum import IntEnum
from typing import Optional, Tuple

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from torch.nn import functional

from champion_league.preprocessors.modules.basemodule import BaseModule
from champion_league.preprocessors.modules.battle_to_tensor import BattleIdx, NB_POKEMON


class MoveIdx(IntEnum):
    move_type = BattleIdx.move_1_type - BattleIdx.move_1_type
    move_acc = BattleIdx.move_1_acc - BattleIdx.move_1_type
    move_base_power = BattleIdx.move_1_base_power - BattleIdx.move_1_type
    move_category = BattleIdx.move_1_category - BattleIdx.move_1_type
    move_drain = BattleIdx.move_1_drain - BattleIdx.move_1_type
    move_heal = BattleIdx.move_1_heal - BattleIdx.move_1_type
    move_pp_ratio = BattleIdx.move_1_pp_ratio - BattleIdx.move_1_type
    move_recoil = BattleIdx.move_1_recoil - BattleIdx.move_1_type


SCALAR_VALUES = [
    MoveIdx.move_acc,
    MoveIdx.move_base_power,
    MoveIdx.move_drain,
    MoveIdx.move_heal,
    MoveIdx.move_pp_ratio,
    MoveIdx.move_recoil,
]


MOVE_MAX = {
    "move_type": 1,
    "move_acc": 1,
    "move_base_power": 1000,
    "move_category": 1,
    "move_drain": 1,
    "move_heal": 1,
    "move_pp_ratio": 1,
    "move_recoil": 1,
}


NB_MOVES = 4


class EmbedMoves(BaseModule):
    def __init__(self, device: Optional[int] = 0):
        self._norm_tensor = torch.tensor([v for _, v in MOVE_MAX.items()]).to(
            f"cuda:{device}"
        )
        self._norm_tensor = self._norm_tensor.expand(NB_POKEMON * NB_MOVES, -1)
        self._output_shape = (
            NB_POKEMON,
            NB_MOVES
            * (
                len(SCALAR_VALUES) + PokemonType.WATER.value + MoveCategory.STATUS.value
            ),
        )

    def embed(self, state: torch.Tensor) -> torch.Tensor:
        b, _, f = state.shape
        moves = torch.reshape(
            state[:, :, BattleIdx.move_1_type :], (b, NB_POKEMON * NB_MOVES, -1)
        )

        scalar_values = moves / self._norm_tensor
        scalar_values = scalar_values[:, :, SCALAR_VALUES]

        move_type = functional.one_hot(
            moves[:, :, MoveIdx.move_type].squeeze(-1).long(),
            num_classes=PokemonType.WATER.value + 1,
        )[:, :, 1:]

        move_category = functional.one_hot(
            moves[:, :, MoveIdx.move_category].squeeze(-1).long(),
            num_classes=MoveCategory.STATUS.value + 1,
        )[:, :, 1:]

        return torch.cat((scalar_values, move_type, move_category), dim=-1).view(
            b, NB_POKEMON, -1
        )

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self._output_shape
