from typing import Dict
from typing import Tuple

import numpy as np
import torch
from poke_env.environment.battle import Battle

from champion_league.preprocessors.base_preprocessor import Preprocessor


class SimplePreprocessor(Preprocessor):
    def embed_battle(self, battle: Battle, **kwargs) -> Dict[str, torch.Tensor]:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        if len(battle.team.values()) > 6 or len(battle.opponent_team.values()) > 6:
            print(len(battle.team.values()), len(battle.opponent_team.values()))

        # Final vector with 10 components
        return {
            "1D": torch.from_numpy(
                np.concatenate(
                    [
                        moves_base_power,
                        moves_dmg_multiplier,
                        [remaining_mon_team, remaining_mon_opponent],
                    ]
                )
            ).flatten()
        }

    @property
    def output_shape(self) -> Tuple[int]:
        return (10,)
