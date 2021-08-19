from enum import auto
from enum import IntEnum

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move


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


def embed_move(battle: Battle, move: Move) -> torch.Tensor:
    embedded_move = torch.zeros(MoveIdx.recoil + 1)
    embedded_move[move.type.value - 1] = 1.0
    embedded_move[MoveIdx.accuracy] = move.accuracy
    embedded_move[MoveIdx.curr_power] = (
        move.type.damage_multiplier(
            battle.opponent_active_pokemon.type_1, battle.opponent_active_pokemon.type_2
        )
        / 4
    )
    embedded_move[MoveIdx.base_power] = move.base_power / 1000
    embedded_move[MoveIdx.is_physical + move.category.value - 1] = 1.0
    embedded_move[MoveIdx.drain] = move.drain
    embedded_move[MoveIdx.heal] = move.heal
    embedded_move[MoveIdx.pp_ratio] = max((move.current_pp / move.max_pp, 0))
    embedded_move[MoveIdx.recoil] = move.recoil
    return embedded_move
