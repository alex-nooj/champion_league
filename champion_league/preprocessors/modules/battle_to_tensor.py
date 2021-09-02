import multiprocessing
from typing import Optional

import torch
from poke_env.environment.battle import Battle
from enum import auto
from enum import IntEnum

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon

from champion_league.utils.abilities import ABILITIES

NB_POKEMON = 12
NB_MOVES = 4
NB_ALLIED_POKEMON = NB_POKEMON // 2
ALLY = 0
ENEMY = 1
ABILITIES_IX = {k: v + 1 for v, k in enumerate(ABILITIES)}


class BattleIdx(IntEnum):
    ability = 0
    health_ratio = auto()
    atk_stat = auto()
    def_stat = auto()
    spa_stat = auto()
    spd_stat = auto()
    spe_stat = auto()
    acc_boost = auto()
    eva_boost = auto()
    atk_boost = auto()
    def_boost = auto()
    spa_boost = auto()
    spd_boost = auto()
    spe_boost = auto()
    status = auto()
    active = auto()
    gender = auto()
    preparing = auto()
    weight = auto()
    type_1 = auto()
    type_2 = auto()
    owner = auto()
    move_1_type = auto()
    move_1_acc = auto()
    move_1_base_power = auto()
    move_1_category = auto()
    move_1_drain = auto()
    move_1_heal = auto()
    move_1_pp_ratio = auto()
    move_1_recoil = auto()
    move_2_type = auto()
    move_2_acc = auto()
    move_2_base_power = auto()
    move_2_category = auto()
    move_2_drain = auto()
    move_2_heal = auto()
    move_2_pp_ratio = auto()
    move_2_recoil = auto()
    move_3_type = auto()
    move_3_acc = auto()
    move_3_base_power = auto()
    move_3_category = auto()
    move_3_drain = auto()
    move_3_heal = auto()
    move_3_pp_ratio = auto()
    move_3_recoil = auto()
    move_4_type = auto()
    move_4_acc = auto()
    move_4_base_power = auto()
    move_4_category = auto()
    move_4_drain = auto()
    move_4_heal = auto()
    move_4_pp_ratio = auto()
    move_4_recoil = auto()


class BattleToTensor:
    def __init__(self, device: Optional[int]):
        self._pokemon_length = BattleIdx.owner + 1
        self._move_length = BattleIdx.move_1_recoil - BattleIdx.move_1_type + 1
        self.device = device

    def copy_move_to_tensor(self, move: Move) -> torch.Tensor:
        move_tensor = torch.zeros(self._move_length)

        move_tensor[BattleIdx.move_1_type - BattleIdx.move_1_type] = move.type.value
        move_tensor[BattleIdx.move_1_acc - BattleIdx.move_1_type] = move.accuracy
        move_tensor[
            BattleIdx.move_1_base_power - BattleIdx.move_1_type
        ] = move.base_power
        move_tensor[
            BattleIdx.move_1_category - BattleIdx.move_1_type
        ] = move.category.value
        move_tensor[BattleIdx.move_1_drain - BattleIdx.move_1_type] = move.drain
        move_tensor[BattleIdx.move_1_heal - BattleIdx.move_1_type] = move.heal
        move_tensor[BattleIdx.move_1_pp_ratio - BattleIdx.move_1_type] = max(
            (move.current_pp / move.max_pp, 0)
        )
        move_tensor[BattleIdx.move_1_recoil - BattleIdx.move_1_type] = move.recoil

        return move_tensor

    def copy_pokemon_to_tensor(self, pokemon: Pokemon, owner: int) -> torch.Tensor:
        pokemon_tensor = torch.zeros(self._pokemon_length)
        if owner == ALLY or pokemon.ability is not None:
            pokemon_tensor[BattleIdx.ability] = ABILITIES_IX[
                pokemon.ability.lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
            ]
        elif (
            pokemon.possible_abilities is not None and "0" in pokemon.possible_abilities
        ):
            pokemon_tensor[BattleIdx.ability] = ABILITIES_IX[
                pokemon.possible_abilities["0"]
                .lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
            ]

        pokemon_tensor[BattleIdx.health_ratio] = pokemon.current_hp_fraction
        for stat_ix, stat in enumerate(["atk", "def", "spa", "spd", "spe"]):
            try:
                pokemon_tensor[BattleIdx.atk_stat + stat_ix] = pokemon.stats[stat]
            except (AttributeError, TypeError):
                pokemon_tensor[BattleIdx.atk_stat + stat_ix] = pokemon.base_stats[stat]

        for stat_ix, stat in enumerate(
            ["accuracy", "evasion", "atk", "def", "spa", "spd", "spe"]
        ):
            pokemon_tensor[BattleIdx.acc_boost + stat_ix] = pokemon.boosts[stat]

        if pokemon.status is not None:
            pokemon_tensor[BattleIdx.status] = pokemon.status.value

        pokemon_tensor[BattleIdx.active] = pokemon.active

        try:
            pokemon_tensor[BattleIdx.gender] = pokemon.gender.value
        except AttributeError:
            pass

        pokemon_tensor[BattleIdx.preparing] = True if pokemon.preparing else False
        pokemon_tensor[BattleIdx.weight] = pokemon.weight
        pokemon_tensor[BattleIdx.type_1] = pokemon.type_1.value
        try:
            pokemon_tensor[BattleIdx.type_2] = pokemon.type_2.value
        except AttributeError:
            pass
        pokemon_tensor[BattleIdx.owner] = owner
        return pokemon_tensor

    def embed(self, battle: Battle) -> torch.Tensor:
        battle_tensor = torch.zeros(NB_POKEMON, len(BattleIdx))

        for poke_ix, (_, pokemon) in enumerate(battle.team.items()):
            battle_tensor[
                poke_ix, : self._pokemon_length
            ] = self.copy_pokemon_to_tensor(pokemon, ALLY)
            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                start_ix = self._pokemon_length + move_ix * self._move_length
                end_ix = self._pokemon_length + (move_ix + 1) * self._move_length
                battle_tensor[poke_ix, start_ix:end_ix] = self.copy_move_to_tensor(move)

        for poke_ix, (_, pokemon) in enumerate(battle.opponent_team.items()):
            battle_tensor[
                poke_ix + NB_ALLIED_POKEMON, : self._pokemon_length
            ] = self.copy_pokemon_to_tensor(pokemon, ENEMY)

            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                start_ix = self._pokemon_length + move_ix * self._move_length
                end_ix = self._pokemon_length + (move_ix + 1) * self._move_length
                battle_tensor[
                    poke_ix + NB_ALLIED_POKEMON, start_ix:end_ix
                ] = self.copy_move_to_tensor(move)

        battle_tensor = battle_tensor.to(f"cuda:{self.device}")
        return battle_tensor.unsqueeze(0)
