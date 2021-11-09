from abc import abstractmethod
from typing import Dict
from typing import Tuple

import torch
from torch import nn

from champion_league.utils.directory_utils import DotDict


class BaseNetwork(nn.Module):
    @abstractmethod
    def forward(
        self, x_internals: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @classmethod
    def from_args(cls, args: DotDict) -> "BaseNetwork":
        raise NotImplementedError

    def reset(self, device: int) -> Dict[str, torch.Tensor]:
        return {}
