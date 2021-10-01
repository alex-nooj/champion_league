from typing import Tuple

import torch

from champion_league.preprocessors.modules.basemodule import BaseModule
from champion_league.preprocessors.modules.battle_to_tensor import BattleIdx
from champion_league.preprocessors.modules.battle_to_tensor import NB_POKEMON


class EmbedAbilities(BaseModule):
    def embed(self, state: torch.Tensor) -> torch.Tensor:
        """Function for embedding the abilities of each pokemon from the state tensor.

        Parameters
        ----------
        state: torch.Tensor
            The current gamestate.

        Returns
        -------
        torch.Tensor
            2-D tensor with size (batch, NB_POKEMON) containing the ability of each pokemon in
            order.
        """
        return state[:, :, BattleIdx.ability].squeeze(-1)

    @property
    def output_shape(self) -> Tuple[int]:
        """The output shape of this preprocessing module.

        Returns
        -------
        Tuple[int]
            Size of this module is 1-dimensional and is equal to the number of pokemon in the
            battle.
        """
        return (NB_POKEMON,)
