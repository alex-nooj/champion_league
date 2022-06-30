from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import Tensor

from champion_league.network.base_network import BaseNetwork
from champion_league.network.modules import Conv
from champion_league.network.modules import Embedding
from champion_league.network.modules import LSTMMod
from champion_league.network.modules.linear import LinearMod
from champion_league.utils.abilities import ABILITIES


class ConvLSTM(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int, ...]],
        embedding_dim: int,
        nb_conv_layers: int,
        nb_channels: int,
        nb_linear_hidden: int,
        nb_linear_layers: int,
        nb_lstm_hidden: int,
        nb_lstm_layer: int,
        nb_action_hidden: int,
        nb_action_layers: int,
        nb_critic_hidden: int,
        nb_critic_layers: int,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 2,
        padding: Optional[int] = 1,
        batch_first: Optional[bool] = True,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()
        seq_len, nb_pokemon, features = in_shape["pokemon"]

        self.abilities_embedding = Embedding(
            in_shape=(int(np.product(in_shape["abilities"])), 1),
            num_embeddings=len(ABILITIES),
            embedding_dim=embedding_dim,
        )

        new_features = features + self.abilities_embedding.output_shape[-1]

        self.conv_net = Conv(
            in_shape=(nb_pokemon, new_features),
            nb_layers=nb_conv_layers,
            nb_channels=nb_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
        )

        self.linears = LinearMod(
            in_shape=(
                int(
                    np.product(
                        self.conv_net.output_shape,
                    )
                ),
            ),
            nb_hidden=nb_linear_hidden,
            nb_layers=nb_linear_layers,
            dropout=dropout,
        )

        self.lstm = LSTMMod(
            in_shape=(seq_len, *self.linears.output_shape),
            nb_hidden=nb_lstm_hidden,
            nb_layer=nb_lstm_layer,
            batch_first=batch_first,
            dropout=dropout,
        )

        self.output = nn.ModuleDict(
            {
                "rough_action": LinearMod(
                    in_shape=self.lstm.output_shape,
                    nb_hidden=nb_action_hidden,
                    nb_layers=nb_action_layers,
                    output_size=nb_actions,
                    dropout=dropout,
                ),
                "critic": LinearMod(
                    in_shape=self.lstm.output_shape,
                    nb_hidden=nb_critic_hidden,
                    nb_layers=nb_critic_layers,
                    output_size=1,
                    dropout=1,
                ),
            }
        )
        self.softmax_layer = nn.Softmax(dim=-1)
        self.tanh_layer = nn.Tanh()

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        b, t, p, f = x["pokemon"].shape

        # Embed the abilities
        abilities = self.abilities_embedding(x["abilities"].view(b * t, p, 1))

        pokemon = torch.cat(
            (
                x["pokemon"].view(b * t, p, f),
                abilities.view(b * t, p, -1),
            ),
            dim=-1,
        )

        conv_out = self.conv_net(pokemon)
        linear_out = self.linears(conv_out.view(b * t, -1))
        lstm_out = self.lstm(linear_out.view(b, t, -1))
        output = {k: v(lstm_out) for k, v in self.output.items()}

        output["action"] = self.softmax_layer(output["rough_action"])
        output["critic"] = self.tanh_layer(output["critic"])
        return output
