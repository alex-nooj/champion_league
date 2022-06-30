from collections import OrderedDict
from typing import Optional
from typing import Tuple

from torch import nn
from torch import Tensor

from champion_league.network.modules.netmod_base import NetMod


class LinearMod(NetMod):
    def __init__(
        self,
        in_shape: Tuple[int],
        nb_hidden: int,
        nb_layers: int,
        output_size: Optional[int] = None,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__(in_shape)
        linears = []
        for i in range(nb_layers):
            linears.append(
                (
                    f"linear_{i}",
                    nn.Linear(
                        in_shape[0] if i == 0 else nb_hidden, nb_hidden, bias=False
                    ),
                )
            )
            linears.append((f"norm_{i}", nn.BatchNorm1d(nb_hidden)))
            linears.append((f"relu_{i}", nn.ReLU()))
            linears.append((f"drop_{i}", nn.Dropout(p=dropout)))
        if output_size is not None:
            linears.append(("output", nn.Linear(nb_hidden, output_size, bias=True)))
        self.linears = nn.Sequential(OrderedDict(linears))
        self.out_shape = nb_hidden

    def forward(self, x: Tensor) -> Tensor:
        return self.linears(x)

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.out_shape,)
