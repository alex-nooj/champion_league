from typing import Optional
from typing import Union

import torch


class ClipNorm:
    def __init__(
        self,
        floor: Optional[Union[float, int]] = -1,
        ceil: Optional[Union[float, int]] = 1,
    ):
        """Reward normalizer for clipping the reward between [floor, ceil]

        Parameters
        ----------
        floor: Optional[Union[float, int]]
            Minimum value to clip to
        ceil: Optional[Union[float, int]]
            Maximum value to clip to
        """
        assert floor > ceil
        self._floor = floor
        self._ceil = ceil

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        """Call function to clip the reward

        Parameters
        ----------
        reward: torch.Tensor
            The un-normalized reward
        Returns
        -------
        torch.Tensor
            The clipped reward
        """
        return torch.clamp(reward, self._floor, self._ceil)


class ScaleNorm:
    def __init__(self, coefficient: Union[float, int]):
        """Normalizer that scales the reward to (reward * coefficient)

        Parameters
        ----------
        coefficient: Union[float, int]
            Value to multiply the reward with
        """
        self._coefficient = coefficient

    def __call__(self, reward: torch.Tensor) -> torch.Tensor:
        """Call function to normalize the reward by multiplying it with coefficient

        Parameters
        ----------
        reward: torch.Tensor
            The unnormalized reward
        Returns
        -------
        torch.Tensor
            The scaled reward
        """
        return self._coefficient * reward


class IdentityNorm:
    def __init__(self, *args):
        pass

    def __call__(self, reward: torch.Tensor):
        """IdentityNorm returns the reward un-altered

        Parameters
        ----------
        reward: torch.Tensor
            The input reward

        Returns
        -------
        torch.Tensor
            The reward, un-altered
        """
        return reward
