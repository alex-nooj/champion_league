from enum import auto
from enum import IntEnum
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status
from torch import Tensor

from champion_league.preprocessor.ops.base_op import Op
from champion_league.preprocessor.util.normalize_stats import normalize_stats


class TeamPokemonIdx(IntEnum):
    damage_for = 0
    damage_against = auto()
    hp_fraction = auto()
    atk_stat = auto()
    def_stat = auto()
    spa_stat = auto()
    spd_stat = auto()
    spe_stat = auto()
    acc_boost = auto()
    eva_boost = auto()
    burn = auto()
    faint = auto()
    freeze = auto()
    paralyze = auto()
    poisoned = auto()
    sleep = auto()
    toxic = auto()


class EmbedTeam(Op):
    def __init__(self, in_shape: Tuple[int]):
        super().__init__(in_shape)
        self._out_shape = (6 * len(TeamPokemonIdx),)

    def preprocess(self, battle: Battle, state: Tensor) -> Tensor:
        team = torch.zeros((6, len(TeamPokemonIdx)))
        opponent = battle.opponent_active_pokemon

        for ix, pokemon in enumerate(battle.team.values()):
            if pokemon.active:
                continue
            # need damage multiplier against opponent
            team[ix, TeamPokemonIdx.damage_for] = max(
                [opponent.damage_multiplier(t) for t in pokemon.types if t is not None]
            )
            # damage multiplier from opponent
            team[ix, TeamPokemonIdx.damage_against] = max(
                [pokemon.damage_multiplier(t) for t in opponent.types if t is not None]
            )
            # stats
            team[
                ix, TeamPokemonIdx.hp_fraction : TeamPokemonIdx.eva_boost + 1
            ] = normalize_stats(pokemon)
            # status
            status = torch.zeros(len(Status))
            if pokemon.status is not None:
                status[pokemon.status.value - 1] = 1.0

            team[ix, TeamPokemonIdx.burn :] = status
        return team.flatten()

    @property
    def output_shape(self) -> Tuple[int]:
        return self._out_shape
