from collections import OrderedDict
from typing import Optional
from typing import Tuple

import numpy as np
from torch import nn
from torch import Tensor

from champion_league.network.modules.netmod_base import NetMod


class Conv(NetMod):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_layers: int,
        nb_channels: int,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 2,
        padding: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__(in_shape)
        s, f = in_shape

        layers = []
        f_out = f
        for i in range(nb_layers):
            layers.append(
                (
                    f"conv_{i}",
                    nn.Conv1d(
                        in_channels=s if i == 0 else nb_channels,
                        out_channels=nb_channels,
                        kernel_size=(kernel_size,),
                        stride=(stride,),
                        padding=(padding,),
                    ),
                )
            )
            layers.append((f"norm_{i}", nn.BatchNorm1d(nb_channels)))
            layers.append((f"relu_{i}", nn.ReLU()))
            layers.append((f"drop_{i}", nn.Dropout(p=dropout)))
            f_out = np.floor(
                ((f_out + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
            )
        self._out_shape = (nb_channels, f_out)
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self._out_shape
