from typing import Tuple

import torch
from adept.utils.util import DotDict
from poke_env.environment.battle import Battle

from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors.modules.embed_allied_pokemon import AlliedPokemonIdx
from champion_league.preprocessors.modules.embed_allied_pokemon import embed_allied_pokemon
from champion_league.preprocessors.modules.embed_enemy_pokemon import embed_enemy_pokemon
from champion_league.preprocessors.modules.embed_move import MoveIdx, embed_move


NB_MOVES = 4


class ModularProcessor(Preprocessor):
    def __init__(self):
        self._output_shape = (12, len(AlliedPokemonIdx) + 4 * len(MoveIdx))

    def embed_battle(self, battle: Battle) -> torch.Tensor:
        embedded_battle = torch.zeros(self._output_shape)
        for poke_ix, (_, pokemon) in enumerate(battle.team.items()):
            embedded_pokemon = embed_allied_pokemon(pokemon)
            embedded_moves = torch.zeros((4, len(MoveIdx)))
            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                embedded_moves[move_ix, :] = embed_move(battle, move)
            embedded_battle[poke_ix, 0 : len(embedded_pokemon)] = torch.clone(embedded_pokemon)
            embedded_battle[poke_ix, len(embedded_pokemon) :] = torch.clone(embedded_moves.view(-1))

        for poke_ix, (_, pokemon) in enumerate(battle.opponent_team.items()):
            embedded_pokemon = embed_enemy_pokemon(pokemon)
            embedded_moves = torch.zeros((4, len(MoveIdx)))
            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                embedded_moves[move_ix, :] = embed_move(battle, move)
            embedded_battle[poke_ix + 6, 0 : len(embedded_pokemon)] = torch.clone(embedded_pokemon)
            embedded_battle[poke_ix + 6, len(embedded_pokemon) :] = torch.clone(
                embedded_moves.view(-1)
            )

        return embedded_battle

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._output_shape


def build_from_args(args: DotDict) -> ModularProcessor:
    return ModularProcessor()
