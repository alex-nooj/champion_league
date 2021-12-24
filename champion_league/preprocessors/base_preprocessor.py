from abc import abstractmethod
from typing import Dict
from typing import Optional
from typing import Tuple

from poke_env.environment.battle import Battle
from torch import Tensor


class Preprocessor:
    """Abstract class for preprocessing."""

    @abstractmethod
    def embed_battle(
        self, battle: Battle, reset: Optional[bool] = False
    ) -> Dict[str, Tensor]:
        """Abstract method for converting a Battle object into a Dictionary of tensors useable by
        the networks.

        Parameters
        ----------
        battle
            The Battle object (game state) to be preprocessed.
        reset
            Whether or not to reset the preprocessing.

        Returns
        -------
        Dict[str, Tensor]
            The state, preprocessed into a form that is useable by the neural network.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Class property describing the preprocessor's output shape.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            The output shape for each head of the preprocessor.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset function."""
        pass
