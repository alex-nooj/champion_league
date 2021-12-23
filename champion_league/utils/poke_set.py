from typing import Dict
from typing import List
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class PokeSet(Dataset):
    def __init__(
        self,
        states: Dict[str, List[Tensor]],
        actions: Tensor,
        rewards: Tensor,
        device: int,
    ):
        """Constructor for a PokeSet

        Parameters
        ----------
        states: Dict[str, List[Tensor]]
            The preprocessed observations from the environment.
        actions: Tensor
            The action chosen by the agent we'd like to imitate.
        rewards: Tensor
            The reward of the current state.
        device: Tensor
            Device to move the tensors to.
        """
        self.device = f"cuda:{device}"
        self._states = states
        self._actions = actions
        self._rewards = rewards

    def __len__(self) -> int:
        """Length of this dataset."""
        return self._actions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """Gets the observation, action, and reward at a given index.

        Parameters
        ----------
        idx: int
            The position of the Tuple we'd like.

        Returns
        -------
        Tuple[Dict[str, Tensor], Tensor, Tensor]
            The state, action, and reward.
        """
        return (
            {k: v[idx].to(self.device).squeeze() for k, v in self._states.items()},
            self._actions[idx].to(self.device).long().squeeze(),
            self._rewards[idx].to(self.device).float().squeeze(),
        )
