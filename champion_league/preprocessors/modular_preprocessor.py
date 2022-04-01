from typing import Dict
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from torch import Tensor

from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors.modules import AlliedPokemonIdx
from champion_league.preprocessors.modules import embed_allied_pokemon
from champion_league.preprocessors.modules import embed_enemy_pokemon
from champion_league.preprocessors.modules import embed_move
from champion_league.preprocessors.modules import MoveIdx
from champion_league.utils.abilities import ABILITIES

NB_POKEMON = 12
NB_MOVES = 4
ABILITIES_IX = {k: v + 1 for v, k in enumerate(ABILITIES)}


class ModularPreprocessor(Preprocessor):
    """Preprocessor for converting Battle objects into tensors"""

    def __init__(self, device: int, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        device
            Which device to move the tensors onto.
        """
        super().__init__(device)
        self._output_shape = {
            "2D": (NB_POKEMON, len(AlliedPokemonIdx) + NB_MOVES * len(MoveIdx)),
            "1D": NB_POKEMON,
        }

    def embed_battle(self, battle: Battle, reset: bool) -> Dict[str, Tensor]:
        """Preprocessing function for this class. It will embed all of the pokemon into a 2D tensor,
        then all of the abilities in a 1D tensor.

        Parameters
        ----------
        battle
            The Battle object (game state) to be preprocessed.
        reset
            Whether or not the battle has been reset

        Returns
        -------
        Dict[str, Tensor]
            The state, preprocessed into a form that is useable by the neural network.
        """
        embedded_battle = torch.zeros(self._output_shape["2D"])
        abilities = torch.zeros(self._output_shape["1D"])

        for poke_ix, (_, pokemon) in enumerate(battle.team.items()):
            embedded_pokemon = embed_allied_pokemon(pokemon)
            embedded_moves = torch.zeros((4, len(MoveIdx)))

            # For each pokemon, embed each move.
            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                embedded_moves[move_ix, :] = embed_move(battle, move)
            embedded_battle[poke_ix, 0 : len(embedded_pokemon)] = torch.clone(
                embedded_pokemon
            )
            embedded_battle[poke_ix, len(embedded_pokemon) :] = torch.clone(
                embedded_moves.view(-1)
            )
            if pokemon.ability is not None:
                abilities[poke_ix] = ABILITIES_IX[
                    pokemon.ability.lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("'", "")
                ]
        for poke_ix, (_, pokemon) in enumerate(battle.opponent_team.items()):
            embedded_pokemon = embed_enemy_pokemon(pokemon)
            embedded_moves = torch.zeros((4, len(MoveIdx)))
            for move_ix, (_, move) in enumerate(pokemon.moves.items()):
                if move_ix == 4:
                    break
                embedded_moves[move_ix, :] = embed_move(battle, move)
            embedded_battle[poke_ix + 6, 0 : len(embedded_pokemon)] = torch.clone(
                embedded_pokemon
            )
            embedded_battle[poke_ix + 6, len(embedded_pokemon) :] = torch.clone(
                embedded_moves.view(-1)
            )
            if pokemon.ability is not None:
                abilities[poke_ix] = ABILITIES_IX[
                    pokemon.ability.lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("'", "")
                ]
            elif (
                pokemon.possible_abilities is not None
                and "0" in pokemon.possible_abilities
            ):
                abilities[poke_ix + NB_POKEMON // 2] = ABILITIES_IX[
                    pokemon.possible_abilities["0"]
                    .lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("'", "")
                ]
        return {
            "2D": embedded_battle.to(self.device).float().unsqueeze(0),
            "1D": abilities.to(self.device).long().unsqueeze(0),
        }

    @property
    def output_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Class property describing the preprocessor's output shape.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            The output shape for each head of the preprocessor.
        """
        return self._output_shape
