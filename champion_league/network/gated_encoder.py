from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from adept.utils.util import DotDict


class Attn(nn.Module):
    def __init__(
        self, in_shape: Tuple[int, int], nb_heads: int, scale: Optional[bool] = True
    ):
        """The attention layer of the transformer encoder.

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        nb_heads: int
            How many input heads to use in the attention layer.
        scale: Optional[bool]
            Option to scale the encoder's attention weights.
        """
        super().__init__()
        _, nb_features = in_shape
        self.nb_heads = nb_heads
        self.mq = nn.Linear(nb_features, nb_features * nb_heads, bias=False)
        self.mk = nn.Linear(nb_features, nb_features * nb_heads, bias=False)
        self.mv = nn.Linear(nb_features, nb_features * nb_heads, bias=False)

        self.projection_output = nn.Sequential(
            OrderedDict(
                [
                    (
                        "projection",
                        nn.Linear(nb_features * nb_heads, nb_features, bias=False),
                    ),
                    ("relu", nn.ReLU()),
                ]
            )
        )
        self.scale = nb_features ** 0.5 if scale else 1.0

    def forward(
        self, q_input: torch.Tensor, kv_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The forward function for the attention layer.

        Parameters
        ----------
        q_input: torch.Tensor
            This is the input that will be passed through the linear layer representing the Q
            matrix.
        kv_input: Optional[torch.Tensor]
            This is the input that will be passed through the linear layers representing the K and
            V matrices. If this attention layer is being in the encoder, then this is the same as
            the kv_input. Otherwise, for the decoder, it is the same as the encoder's output.

        Returns
        -------
        torch.Tensor
            The output of the attention layer.
        """
        b, q_s, f = q_input.shape
        if kv_input is None:
            kv_input = torch.clone(q_input)

        _, kv_s, _ = kv_input.shape

        q_input = q_input.view(b * q_s, f)
        k_input = torch.clone(kv_input).view(b * kv_s, f)
        v_input = torch.clone(kv_input).view(b * kv_s, f)

        q = self.mq(q_input).view(b, q_s, f, self.nb_heads)
        k = self.mk(k_input).view(b, kv_s, f, self.nb_heads)
        v = self.mv(v_input).view(b, kv_s, f, self.nb_heads)

        qkt = torch.einsum("bifh,bjfh->bijh", q, k)

        attn_weights = (qkt * self.scale).sigmoid()
        weighted_v = torch.einsum("bijh,bjfh->bifh", attn_weights, v)
        weighted_v = torch.reshape(weighted_v, (b * q_s, f * self.nb_heads))

        return self.projection_output(weighted_v).view(b, q_s, f)


class Gate(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        """The gating layer for the encoder.

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        """
        super().__init__()
        # self.gru = nn.GRU(input_size=in_shape[1], hidden_size=in_shape[1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        b, s, f = x.shape

        # gate_output, _ = self.gru(
        #     torch.reshape(y, (1, b * s, f)),
        #     torch.reshape(x, (1, b * s, f)),
        # )

        # return gate_output.view(b, s, f)
        return x + y


class LNorm(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        """

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        b, s, f = x.shape

        return self.layer_norm(torch.reshape(x, (b * s, f))).view(b, s, f)


class Encoder(nn.Module):
    def __init__(
        self, in_shape: Tuple[int, int], nb_heads: int, scale: Optional[bool] = True
    ):
        """

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        nb_heads
        scale
        """

        super().__init__()
        self.first_block = nn.Sequential(
            OrderedDict(
                [
                    ("first_norm", nn.LayerNorm(in_shape[1])),
                    ("attn_layer", Attn(in_shape, nb_heads, scale)),
                ]
            )
        )
        self.first_gate = Gate(in_shape)
        self.second_block = nn.Sequential(
            OrderedDict(
                [
                    ("second_norm", nn.LayerNorm(in_shape[1])),
                    (
                        "projection_layer",
                        nn.Linear(in_shape[1], in_shape[1], bias=False),
                    ),
                    ("projection_relu", nn.ReLU()),
                ]
            )
        )
        self.second_gate = Gate(in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x

        Returns
        -------

        """

        attn_out = self.first_block(x)
        gate_out = self.first_gate(x, attn_out)
        norm_out = self.second_block(gate_out)
        return self.second_gate(gate_out, norm_out)


class GatedEncoder(nn.Module):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Tuple[int, int],
        nb_encoders: Optional[int],
        nb_heads: Optional[int],
        nb_layers: Optional[int],
        scale: Optional[bool],
        dropout: Optional[float],
    ):
        """

        Parameters
        ----------
        nb_actions
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size.
        nb_encoders
        nb_heads
        nb_layers
        scale
        dropout
        """
        super().__init__()

        if nb_encoders is None:
            nb_encoders = 1
        if nb_heads is None:
            nb_heads = 1
        if nb_layers is None:
            nb_layers = 3
        if scale is None:
            scale = False

        encoders = [
            (f"encoder_{i}", Encoder(in_shape, nb_heads, scale=scale))
            for i in range(nb_encoders)
        ]
        avg_pooling = [
            ("avg_pooling", nn.AvgPool2d((in_shape[0], 1))),
            ("flatten", nn.Flatten()),
        ]

        linears = []
        for i in range(nb_layers):
            linears.append(
                (f"linear_{i}", nn.Linear(in_shape[1], in_shape[1], bias=False))
            )
            linears.append((f"norm_{i}", nn.BatchNorm1d(in_shape[1])))
            linears.append((f"relu_{i}", nn.ReLU()))

        self.encoders = nn.Sequential(OrderedDict(encoders + avg_pooling + linears))

        self.output_layers = nn.ModuleDict(
            {
                "action": nn.Sequential(
                    OrderedDict(
                        [
                            ("action_1", nn.Linear(in_shape[1], nb_actions, bias=True)),
                        ]
                    )
                ),
                "critic": nn.Sequential(
                    OrderedDict([("critic_1", nn.Linear(in_shape[1], 1, bias=True))])
                ),
            }
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoders(x)
        rough_action = self.output_layers["action"](encoder_out)
        critic = self.output_layers["critic"](encoder_out)
        soft_action = self.softmax_layer(rough_action)
        return {
            "action": soft_action,
            "critic": critic,
            "rough_action": rough_action,
        }


def build_from_args(args: DotDict) -> GatedEncoder:
    return GatedEncoder(
        args.nb_actions,
        args.in_shape,
        args.nb_encoders,
        args.nb_heads,
        args.nb_layers,
        args.scale,
        args.dropout,
    )
