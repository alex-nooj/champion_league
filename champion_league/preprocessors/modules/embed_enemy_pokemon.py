from enum import auto
from enum import IntEnum

import torch
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.status import Status

from champion_league.utils.abilities import ABILITIES


class EnemyPokemonIdx(IntEnum):
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
    atk_base = auto()
    def_base = auto()
    spa_base = auto()
    spd_base = auto()
    spe_base = auto()
    acc_boost = auto()
    eva_boost = auto()
    atk_boost = auto()
    def_boost = auto()
    spa_boost = auto()
    spd_boost = auto()
    spe_boost = auto()
    ability_bit0 = auto()
    ability_bit1 = auto()
    ability_bit2 = auto()
    ability_bit3 = auto()
    ability_bit4 = auto()
    ability_bit5 = auto()
    ability_bit6 = auto()
    ability_bit7 = auto()
    ability_bit8 = auto()
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


def embed_enemy_pokemon(pokemon: Pokemon) -> torch.Tensor:
    """Function for converting an enemy's pokemon into a tensor

    Parameters
    ----------
    pokemon: Pokemon
        The Showdown! styled pokemon to be converted.

    Returns
    -------
    torch.Tensor
        The pokemon as a tensor.
    """
    embedded_pokemon = torch.zeros(EnemyPokemonIdx.toxicked + 1)

    if pokemon.ability is not None:
        embedded_pokemon[
            EnemyPokemonIdx.ability_bit0 : EnemyPokemonIdx.ability_bit8 + 1
        ] = torch.tensor(
            ABILITIES[
                pokemon.ability.lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
            ]
        )
    else:
        embedded_pokemon[
            EnemyPokemonIdx.ability_bit0 : EnemyPokemonIdx.ability_bit8 + 1
        ] = -1

    embedded_pokemon[EnemyPokemonIdx.health_ratio] = pokemon.current_hp_fraction
    embedded_pokemon[EnemyPokemonIdx.atk_base] = (
        pokemon.base_stats["atk"] / 190.0
    )  # Mega Mewtwo X
    embedded_pokemon[EnemyPokemonIdx.def_base] = (
        pokemon.base_stats["def"] / 250.0
    )  # Eternatus
    embedded_pokemon[EnemyPokemonIdx.spa_base] = (
        pokemon.base_stats["spa"] / 194.0
    )  # Mega Mewtwo Y
    embedded_pokemon[EnemyPokemonIdx.spd_base] = (
        pokemon.base_stats["spd"] / 250.0
    )  # Eternatus
    embedded_pokemon[EnemyPokemonIdx.spe_base] = (
        pokemon.base_stats["spe"] / 200.0
    )  # Regieleki

    embedded_pokemon[EnemyPokemonIdx.acc_boost] = pokemon.boosts["accuracy"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.eva_boost] = pokemon.boosts["evasion"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.atk_boost] = pokemon.boosts["atk"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.def_boost] = pokemon.boosts["def"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.spa_boost] = pokemon.boosts["spa"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.spd_boost] = pokemon.boosts["spd"] / 6.0
    embedded_pokemon[EnemyPokemonIdx.spe_boost] = pokemon.boosts["spe"] / 6.0

    if pokemon.status is not None:
        status = torch.zeros(len(Status))
        status[pokemon.status.value - 1] = 1.0
        embedded_pokemon[EnemyPokemonIdx.burned :] = status
    embedded_pokemon[EnemyPokemonIdx.active] = pokemon.active
    try:
        embedded_pokemon[EnemyPokemonIdx.female + pokemon.gender.value - 1] = 1.0
    except AttributeError:
        pass
    embedded_pokemon[EnemyPokemonIdx.preparing] = True if pokemon.preparing else False
    embedded_pokemon[EnemyPokemonIdx.weight] = pokemon.weight / 2300

    embedded_pokemon[pokemon.type_1.value - 1] = 1.0
    if pokemon.type_2 is not None:
        embedded_pokemon[pokemon.type_2.value - 1] = 1.0

    return embedded_pokemon
