from enum import auto
from enum import IntEnum

import torch
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status

from champion_league.utils.abilities import ABILITIES


class AlliedPokemonIdx(IntEnum):
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
    active = auto()
    female = auto()
    male = auto()
    neutral = auto()
    item = auto()
    preparing = auto()
    weight = auto()
    burned = auto()
    fainted = auto()
    frozen = auto()
    paralyzed = auto()
    poisoned = auto()
    sleeping = auto()
    toxicked = auto()


def embed_allied_pokemon(pokemon: Pokemon) -> torch.Tensor:
    embedded_pokemon = torch.zeros(AlliedPokemonIdx.toxicked + 1)

    embedded_pokemon[AlliedPokemonIdx.health_ratio] = pokemon.current_hp_fraction
    embedded_pokemon[AlliedPokemonIdx.atk_stat] = (
        pokemon.stats["atk"] / 526.0
    )  # Mega Mewtwo X
    embedded_pokemon[AlliedPokemonIdx.def_stat] = (
        pokemon.stats["def"] / 614.0
    )  # Eternatus
    embedded_pokemon[AlliedPokemonIdx.spa_stat] = (
        pokemon.stats["spa"] / 526.0
    )  # Mega Mewtwo Y
    embedded_pokemon[AlliedPokemonIdx.spd_stat] = (
        pokemon.stats["spd"] / 614.0
    )  # Eternatus
    embedded_pokemon[AlliedPokemonIdx.spe_stat] = (
        pokemon.stats["spe"] / 504.0
    )  # Regieleki

    embedded_pokemon[AlliedPokemonIdx.acc_boost] = pokemon.boosts["accuracy"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.eva_boost] = pokemon.boosts["evasion"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.atk_boost] = pokemon.boosts["atk"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.def_boost] = pokemon.boosts["def"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.spa_boost] = pokemon.boosts["spa"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.spd_boost] = pokemon.boosts["spd"] / 6.0
    embedded_pokemon[AlliedPokemonIdx.spe_boost] = pokemon.boosts["spe"] / 6.0

    if pokemon.status is not None:
        status = torch.zeros(len(Status))
        status[pokemon.status.value - 1] = 1.0
        embedded_pokemon[AlliedPokemonIdx.burned :] = status
    embedded_pokemon[AlliedPokemonIdx.active] = pokemon.active
    try:
        embedded_pokemon[AlliedPokemonIdx.female + pokemon.gender.value - 1] = 1.0
    except AttributeError:
        pass
    embedded_pokemon[AlliedPokemonIdx.preparing] = True if pokemon.preparing else False
    embedded_pokemon[AlliedPokemonIdx.weight] = pokemon.weight / 2300

    embedded_pokemon[pokemon.type_1.value - 1] = 1.0
    if pokemon.type_2 is not None:
        embedded_pokemon[pokemon.type_2.value - 1] = 1.0

    return embedded_pokemon
