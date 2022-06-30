from typing import Optional
from typing import Tuple

from torch import nn
from torch import Tensor

from champion_league.network.modules.netmod_base import NetMod


class LSTMMod(NetMod):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_hidden: int,
        nb_layer: int,
        batch_first: Optional[bool] = True,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__(in_shape)
        sequence_len, nb_features = in_shape
        self.lstm = nn.LSTM(
            input_size=nb_features,
            hidden_size=nb_hidden,
            num_layers=nb_layer,
            batch_first=batch_first,
            dropout=dropout,
        )

        self._out_shape = nb_hidden

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.lstm(x)
        return y[:, -1, :]

    @property
    def output_shape(self) -> Tuple[int]:
        return (self._out_shape,)
