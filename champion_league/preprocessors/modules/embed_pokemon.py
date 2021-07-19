import torch
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status

from champion_league.preprocessors.modules.basemodule import BaseModule
from champion_league.utils.abilities import ABILITIES


class EmbedPokemon(BaseModule):
    def embed(self, pokemon: Pokemon, ally: bool) -> torch.Tensor:
        pokemon_type = torch.zeros(len(PokemonType))
        pokemon_type[pokemon.type_1.value] = 1.0

        if pokemon.type_2 is not None:
            pokemon_type[pokemon.type_2.value] = 1.0

        if pokemon.ability is not None:
            ability = ABILITIES[
                pokemon.ability.lower()
                .replace(" ", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
                .replace("'", "")
            ]
        else:
            ability = 0.0

        hp_ratio = pokemon.current_hp_fraction
        base_hp = pokemon.base_stats["hp"] / 255.0
        base_atk = pokemon.base_stats["atk"] / 190.0
        base_def = pokemon.base_stats["def"] / 250.0
        base_spa = pokemon.base_stats["spa"] / 194.0
        base_spd = pokemon.base_stats["spd"] / 250.0
        base_spe = pokemon.base_stats["spe"] / 200.0

        acc_boost = pokemon.boosts["accuracy"] / 4.0
        eva_boost = pokemon.boosts["evasion"] / 4.0
        atk_boost = pokemon.boosts["atk"] / 4.0
        def_boost = pokemon.boosts["def"] / 4.0
        spa_boost = pokemon.boosts["spa"] / 4.0
        spd_boost = pokemon.boosts["spd"] / 4.0
        spe_boost = pokemon.boosts["spe"] / 4.0

        status = torch.zeros(len(Status))
        if pokemon.status is not None:
            status[pokemon.status.value] = 1.0

        active = pokemon.active

        return torch.cat(
            (
                pokemon_type,
                torch.tensor(
                    [
                        ability,
                        hp_ratio,
                        base_hp,
                        base_atk,
                        base_def,
                        base_spa,
                        base_spd,
                        base_spe,
                        acc_boost,
                        eva_boost,
                        atk_boost,
                        def_boost,
                        spa_boost,
                        spd_boost,
                        spe_boost,
                        active,
                    ]
                ),
                status,
            )
        )
