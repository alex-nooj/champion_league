from enum import auto
from enum import IntEnum
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from torch import Tensor

from champion_league.preprocessor.ops.base_op import Op


class MoveIdx(IntEnum):
    bug_type = 0
    dark_type = auto()
    dragon_type = auto()
    electric_type = auto()
    fairy_type = auto()
    fighting_type = auto()
    fire_type = auto()
    flying_type = auto()
    ghost_type = auto()
    grass_type = auto()
    ground_type = auto()
    ice_type = auto()
    normal_type = auto()
    poison_type = auto()
    psychic_type = auto()
    rock_type = auto()
    steel_type = auto()
    water_type = auto()
    accuracy = auto()
    base_power = auto()
    curr_power = auto()
    is_physical = auto()
    is_special = auto()
    is_status = auto()
    drain = auto()
    heal = auto()
    pp_ratio = auto()
    recoil = auto()
    damage_multiplier = auto()


class EmbedMoves(Op):
    def __init__(self, in_shape: Tuple[int], ally: bool):
        super().__init__(in_shape)
        self.ally = ally
        self._out_shape = (in_shape[0] + 4 * len(MoveIdx),)

    def preprocess(self, battle: Battle, state: Tensor) -> Tensor:
        if self.ally:
            pokemon = battle.active_pokemon
        else:
            pokemon = battle.opponent_active_pokemon

        pokemon_moves = list(pokemon.moves.values())
        if len(pokemon_moves) > 4:
            pokemon_moves = pokemon_moves[:4]

        moves = torch.stack(
            [self._move_to_tensor(move, battle) for move in pokemon_moves]
        ).view(-1)

        ret_tensor = torch.zeros(self._out_shape)
        ret_tensor[: self._in_shape[0]] = state
        ret_tensor[self._in_shape[0] :] = moves
        return ret_tensor

    def _move_to_tensor(self, move: Move, battle: Battle) -> Tensor:
        move_tensor = torch.zeros(len(MoveIdx))
        move_tensor[move.type.value - 1] = 1.0
        move_tensor[MoveIdx.accuracy] = move.accuracy
        move_tensor[MoveIdx.curr_power] = (
            move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
            )
            / 4
        )
        move_tensor[MoveIdx.base_power] = move.base_power / 1000
        move_tensor[MoveIdx.is_physical + move.category.value - 1] = 1.0
        move_tensor[MoveIdx.drain] = move.drain
        move_tensor[MoveIdx.heal] = move.heal
        move_tensor[MoveIdx.pp_ratio] = max((move.current_pp, move.max_pp, 0))
        move_tensor[MoveIdx.recoil] = move.recoil
        if self.ally:
            move_tensor[
                MoveIdx.damage_multiplier
            ] = battle.opponent_active_pokemon.damage_multiplier(move)
        else:
            move_tensor[
                MoveIdx.damage_multiplier
            ] = battle.active_pokemon.damage_multiplier(move)
        return move_tensor

    @property
    def output_shape(self) -> Tuple[int]:
        return self._out_shape
