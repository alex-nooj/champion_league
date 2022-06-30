import torch
from poke_env.environment.pokemon import Pokemon
from torch import Tensor

from champion_league.preprocessor.util.stat_estimation import stat_estimation


def normalize_stats(pokemon: Pokemon) -> Tensor:
    hp_fraction = pokemon.current_hp_fraction
    atk_stat = stat_estimation(pokemon, "atk") / 526.0  # Mega Mewtwo X
    def_stat = stat_estimation(pokemon, "def") / 614.0  # Eternatus
    spa_stat = stat_estimation(pokemon, "spa") / 526.0  # Mega Mewtwo Y
    spd_stat = stat_estimation(pokemon, "spd") / 614.0  # Eternatus
    spe_stat = stat_estimation(pokemon, "spe") / 504.0  # Regieleki
    acc_boost = pokemon.boosts["accuracy"] / 6.0
    eva_boost = pokemon.boosts["evasion"] / 6.0
    return torch.tensor(
        [
            hp_fraction,
            atk_stat,
            def_stat,
            spa_stat,
            spd_stat,
            spe_stat,
            acc_boost,
            eva_boost,
        ]
    )
