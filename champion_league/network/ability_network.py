from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from champion_league.network.base_network import BaseNetwork
from champion_league.network.gated_encoder import GatedEncoder
from champion_league.network.modules import Embedding
from champion_league.utils.abilities import ABILITIES


class AbilityNetwork(BaseNetwork):
    """Network that embeds pokemon abilities and passes them (and the pokemon) into an encoder."""

    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int, ...]],
        embedding_dim: Optional[int],
        nb_encoders: Optional[int],
        nb_heads: Optional[int],
        nb_layers: Optional[int],
        scale: Optional[bool],
        dropout: Optional[float],
    ):
        """Constructor for the neural network.

        Parameters
        ----------
        nb_actions
            The size of the action space.
        in_shape
            The size of the observation space for each input head.
        embedding_dim
            The size of the abilities after the embedding.
        nb_encoders
            How many encoders to stack together.
        nb_heads
            How many heads to use in the multihead attention layer.
        nb_layers
            The number of linear layers to append after the encoders.
        scale
            Whether to scale the attention within the encoder.
        dropout
            The percentage for the dropout layers.
        """
        super().__init__()

        self.abilities_embedding = Embedding(
            in_shape=in_shape["abilities"],
            num_embeddings=len(ABILITIES),
            embedding_dim=embedding_dim,
        )

        self.gated_encoder = GatedEncoder(
            nb_actions=nb_actions,
            in_shape={
                "pokemon": (
                    in_shape["pokemon"][0],
                    in_shape["pokemon"][1] + embedding_dim,
                )
            },
            nb_encoders=nb_encoders,
            nb_heads=nb_heads,
            nb_layers=nb_layers,
            scale=scale,
            dropout=dropout,
        )

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward function for the nerual network

        Parameters
        ----------
        x
            The current observation.

        Returns
        -------
        Dict[str, Tensor]
            The output predictions. The output predictions contain keys 'rough action' for the
            action prediction sans softmax, 'action' for action prediction with softmax, and
            'critic' for the critic prediction.
        """

        abilities = self.abilities_embedding(x["abilities"])
        return self.gated_encoder(
            x={
                "pokemon": torch.cat(
                    (x["pokemon"], abilities),
                    dim=-1,
                )
            }
        )
