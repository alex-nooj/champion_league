from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from champion_league.network.base_network import BaseNetwork
from champion_league.network.gated_encoder import GatedEncoder
from champion_league.utils.abilities import ABILITIES
from champion_league.utils.directory_utils import DotDict


class AbilityNetwork(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int, int]],
        embedding_dim: Optional[int],
        nb_encoders: Optional[int],
        nb_heads: Optional[int],
        nb_layers: Optional[int],
        scale: Optional[bool],
    ):
        super().__init__()

        self.abilities_embedding = nn.Embedding(
            num_embeddings=len(ABILITIES),
            embedding_dim=embedding_dim,
        )

        self.gated_encoder = GatedEncoder(
            nb_actions=nb_actions,
            in_shape={
                "2D": (
                    in_shape["2D"][0],
                    in_shape["2D"][1] + embedding_dim,
                )
            },
            nb_encoders=nb_encoders,
            nb_heads=nb_heads,
            nb_layers=nb_layers,
            scale=scale,
        )

    def forward(
        self, x_internals: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.tensor], Dict[str, torch.Tensor]]:
        x = x_internals["x"]

        abilities = self.abilities_embedding(x["1D"])
        return self.gated_encoder(
            x_internals={
                "x": {
                    "2D": torch.cat(
                        (x["2D"], abilities),
                        dim=-1,
                    )
                }
            }
        )

    @classmethod
    def from_args(cls, args: DotDict) -> "AbilityNetwork":
        return AbilityNetwork(
            nb_actions=args.nb_actions,
            in_shape=args.in_shape,
            embedding_dim=args.embedding_dim or 5,
            nb_encoders=args.nb_encoders or 1,
            nb_heads=args.nb_heads or 1,
            nb_layers=args.nb_layers or 3,
            scale=args.scale or False,
        )
