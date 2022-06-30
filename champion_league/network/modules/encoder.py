from collections import OrderedDict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from champion_league.network.modules.netmod_base import NetMod


class Encoder(NetMod):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_heads: int,
        scale: Optional[bool] = True,
        dropout: Optional[float] = 0.0,
    ):
        """A gated encoder, as described in the Stablizing the Transformer for RL paper.

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        nb_heads: int
            How many heads to use for the multi-head attention layer.
        scale: Optional[bool]
            Whether to scale the input down.
        dropout: Optional[float]
            Dropout to use in the network
        """

        super().__init__(in_shape)
        self.first_block = nn.Sequential(
            OrderedDict(
                [
                    ("first_norm", LNorm(in_shape)),
                    ("attn_layer", Attn(in_shape, nb_heads, scale)),
                    ("dropout", nn.Dropout(p=dropout)),
                ]
            )
        )
        self.second_block = nn.Sequential(
            OrderedDict(
                [
                    ("second_norm", LNorm(in_shape)),
                    (
                        "projection_layer",
                        Projection(in_shape),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function for the gated encoder.

        Parameters
        ----------
        x: torch.Tensor
            The input to the gated encoder, with shape [batch_size, *in_shape]

        Returns
        -------
        torch.Tensor
            The output of the gated encoder.
        """

        attn_out = self.first_block(x)
        gate_out = x + attn_out
        norm_out = self.second_block(gate_out)
        return gate_out + norm_out

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.input_shape


class Attn(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_heads: int,
        scale: Optional[bool] = True,
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


class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self._dim)


class LNorm(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        """Layer norm module for the Encoder.

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape of the data. This should not include batch size and should be the same
            shape as the input to the overall encoder.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the layernorm module.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape [*, *in_shape].

        Returns
        -------
        torch.Tensor
            The normalized output.
        """
        b, s, f = x.shape

        return self.layer_norm(torch.reshape(x, (b * s, f))).view(b, s, f)


class Projection(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        super().__init__()
        self.projection_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "projection_layer",
                        nn.Linear(in_shape[1], in_shape[1], bias=False),
                    ),
                    ("projection_relu", nn.ReLU()),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        proj_x = self.projection_layer(x.view(b * s, f))
        return proj_x.view(b, s, f)
