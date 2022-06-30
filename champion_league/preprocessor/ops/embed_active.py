from enum import auto
from enum import IntEnum
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status
from torch import Tensor

from champion_league.preprocessor.ops.base_op import Op
from champion_league.preprocessor.util.normalize_stats import normalize_stats


class ActiveObsIdx(IntEnum):
    hp_fraction = 0
    atk_stat = auto()
    def_stat = auto()
    spa_stat = auto()
    spd_stat = auto()
    spe_stat = auto()
    acc_boost = auto()
    eva_boost = auto()
    bug = auto()
    dark = auto()
    dragon = auto()
    electric = auto()
    fairy = auto()
    fighting = auto()
    fire = auto()
    flying = auto()
    ghost = auto()
    grass = auto()
    ground = auto()
    ice = auto()
    normal = auto()
    poison = auto()
    psychic = auto()
    rock = auto()
    steel = auto()
    water = auto()
    burn = auto()
    faint = auto()
    freeze = auto()
    paralyze = auto()
    poisoned = auto()
    sleep = auto()
    toxic = auto()


class EmbedActive(Op):
    def __init__(self, in_shape: Tuple[int], ally: bool):
        super().__init__(in_shape)
        self.ally = ally
        self._out_shape = (len(ActiveObsIdx),)

    def preprocess(self, battle: Battle, state: Tensor) -> Tensor:
        if self.ally:
            pokemon = battle.active_pokemon
        else:
            pokemon = battle.opponent_active_pokemon

        stats = normalize_stats(pokemon)

        status = torch.zeros(len(Status))
        if pokemon.status is not None:
            status[pokemon.status.value - 1] = 1.0
        poke_types = torch.zeros(18)
        poke_types[pokemon.type_1.value - 1] = 1.0
        if pokemon.type_2 is not None:
            poke_types[pokemon.type_2.value - 1] = 1.0

        return torch.cat((stats, poke_types, status))

    @property
    def output_shape(self) -> Tuple[int]:
        return self._out_shape
