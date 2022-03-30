from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from champion_league.network.base_network import BaseNetwork
from champion_league.network.gated_encoder import Encoder
from champion_league.network.gated_encoder import Squeeze
from champion_league.utils.abilities import ABILITIES


class EncoderLSTM(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int, int]],
        embedding_dim: Optional[int],
        nb_encoders: Optional[int],
        nb_heads: Optional[int],
        nb_layers: Optional[int],
        scale: Optional[bool],
        lstm_hidden: Optional[int],
        linear_hidden: Optional[int],
    ):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = 5
        if nb_encoders is None:
            nb_encoders = 1
        if nb_heads is None:
            nb_heads = 1
        if nb_layers is None:
            nb_layers = 1
        if scale is None:
            scale = False
        if lstm_hidden is None:
            lstm_hidden = 512
        if linear_hidden is None:
            linear_hidden = 128

        self.lstm_hidden = lstm_hidden

        self.abilities_embedding = nn.Embedding(
            num_embeddings=len(ABILITIES),
            embedding_dim=embedding_dim,
        )

        encoders = [
            (
                f"encoder_{i}",
                Encoder(
                    in_shape=(in_shape["2D"][0], in_shape["2D"][1] + embedding_dim),
                    nb_heads=nb_heads,
                    scale=scale,
                ),
            )
            for i in range(nb_encoders)
        ]

        avg_pooling = [
            ("avg_pooling", nn.AvgPool2d((in_shape["2D"][0], 1))),
            ("flatten", nn.Flatten()),
            ("squeeze", Squeeze(dim=1)),
        ]

        self.encoders = nn.Sequential(OrderedDict(encoders + avg_pooling))

        self.lstm = nn.LSTMCell(
            input_size=in_shape["2D"][1] + embedding_dim,
            hidden_size=lstm_hidden,
        )

        linears = []
        for i in range(nb_layers):
            linears.append(
                (
                    f"linear_{i}",
                    nn.Linear(
                        lstm_hidden if i == 0 else linear_hidden,
                        linear_hidden,
                        bias=False,
                    ),
                )
            )
            linears.append((f"norm_{i}", nn.BatchNorm1d(linear_hidden)))
            linears.append((f"relu_{i}", nn.ReLU()))

        self.linears = nn.Sequential(OrderedDict(linears))

        self.output_layers = nn.ModuleDict(
            {
                "action": nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "action_1",
                                nn.Linear(linear_hidden, nb_actions, bias=True),
                            ),
                        ]
                    )
                ),
                "critic": nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "critic_1",
                                nn.Linear(linear_hidden, 1, bias=True),
                            )
                        ]
                    )
                ),
            }
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(
        self, x_internals: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x = x_internals["x"]
        internals = x_internals["internals"]

        b, s, f = x["2D"].shape
        abilities = self.abilities_embedding(x["1D"])
        encoder_input = torch.zeros(
            (b, s, f + abilities.shape[-1]), device=x["2D"].device
        )
        encoder_input[:, :, :f] += x["2D"]
        encoder_input[:, :, f:] += abilities
        encoder_out = self.encoders(encoder_input)
        hx, cx = self.lstm(encoder_out, (internals["hx"], internals["cx"]))

        lin_out = self.linears(hx)
        rough_action = self.output_layers["action"](lin_out)
        soft_action = self.softmax_layer(rough_action)
        critic = self.output_layers["critic"](lin_out)

        return (
            {
                "action": soft_action,
                "critic": critic,
                "rough_action": rough_action,
            },
            {"hx": torch.clone(hx).detach(), "cx": torch.clone(cx).detach()},
        )

    def reset(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "hx": torch.zeros((1, self.lstm_hidden), device=device),
            "cx": torch.zeros((1, self.lstm_hidden), device=device),
        }
