from typing import Tuple

from torch import nn
from torch import Tensor


class NetMod(nn.Module):
    def __init__(self, in_shape: Tuple[int, ...]):
        super().__init__()
        self.in_shape = in_shape

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError
