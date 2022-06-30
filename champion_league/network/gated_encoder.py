from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from champion_league.config.load_configs import get_default_args
from champion_league.network.base_network import BaseNetwork
from champion_league.network.modules.encoder import Encoder
from champion_league.network.modules.linear import LinearMod


class GatedEncoder(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int, int]],
        nb_encoders: Optional[int] = None,
        nb_heads: Optional[int] = None,
        nb_layers: Optional[int] = None,
        scale: Optional[bool] = None,
        dropout: Optional[float] = None,
    ):
        """The Multi-head, Multi-Encoder, Order-Invariant Gated Encoder module.

        Parameters
        ----------
        nb_actions: int
            The size of the action space.
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size.
        nb_encoders: Optional[int]
            How many encoders to use sequentially.
        nb_heads: Optional[int]
            How many heads to use in every multi-head attention layer.
        nb_layers: Optional[int]
            How many linear layers to include after the average pooling layer.
        scale: Optional[bool]
            Whether to scale the input to the encoders.
        dropout: Optional[float]
            Dropout to use in the network
        """
        super().__init__()
        cfg = get_default_args(__file__)
        nb_encoders = nb_encoders or cfg.nb_encoders
        nb_heads = nb_heads or cfg.nb_heads
        nb_layers = nb_layers or cfg.nb_layers
        scale = scale or cfg.scale
        dropout = dropout or cfg.dropout

        encoders = [
            (
                f"encoder_{i}",
                Encoder(in_shape["2D"], nb_heads, scale=scale, dropout=dropout),
            )
            for i in range(nb_encoders)
        ]

        linears = LinearMod((in_shape["2D"][1],), in_shape["2D"][1], nb_layers)

        self.encoders = nn.Sequential(
            OrderedDict(
                [
                    ("encoder", encoders),
                    ("avg_pooling", nn.AvgPool2d((in_shape["2D"][0], 1))),
                    ("flatten", nn.Flatten()),
                    ("linears", linears),
                ]
            )
        )

        self.output_layers = nn.ModuleDict(
            {
                "action": nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "action_1",
                                nn.Linear(
                                    linears.output_shape[0], nb_actions, bias=True
                                ),
                            ),
                        ]
                    )
                ),
                "critic": nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "critic_1",
                                nn.Linear(linears.output_shape[0], 1, bias=True),
                            ),
                            (
                                "critic_tanh",
                                torch.nn.Tanh(),
                            ),
                        ]
                    )
                ),
            }
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """The forward function for the Gated Encoder.

        Args:
            x: The input state to the network. Must contain the key 'pokemon'

        Returns:
            Dict[str, torch.Tensor]: The output predictions of the network. These predictions
            include the critic value ('critic'), the softmaxed action distribution ('action'), and
            the non-softmaxed output for the action distribution ('rough_action').
        """
        encoder_out = self.encoders(x["pokemon"])
        rough_action = self.output_layers["action"](encoder_out)
        critic = self.output_layers["critic"](encoder_out)
        soft_action = self.softmax_layer(rough_action)
        return {
            "action": soft_action,
            "critic": critic,
            "rough_action": rough_action,
        }
