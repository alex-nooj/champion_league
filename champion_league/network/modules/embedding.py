from typing import Tuple

from torch import nn
from torch import Tensor

from champion_league.network.modules.netmod_base import NetMod


class Embedding(NetMod):
    def __init__(
        self, in_shape: Tuple[int, ...], num_embeddings: int, embedding_dim: int
    ):
        super().__init__(in_shape)
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self._out_shape = (*in_shape[:-1], embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x.long())

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self._out_shape
