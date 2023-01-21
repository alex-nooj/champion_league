import typing
from collections import OrderedDict
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from champion_league.config.load_configs import get_default_args
from champion_league.network.base_network import BaseNetwork


class SimpleNetwork(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int]],
        layers: typing.Optional[typing.List[int]] = None,
        critic_hidden: Optional[int] = None,
        action_hidden: Optional[int] = None,
    ):
        super().__init__()
        cfg = get_default_args(__file__)
        layers = layers or cfg.layers
        critic_hidden = critic_hidden or cfg.critic_hidden
        action_hidden = action_hidden or cfg.action_hidden

        total_in_shape = sum([np.prod(v) for v in in_shape.values()])

        in_sizes = [total_in_shape] + layers[:-1]
        linears = []
        for ix, (in_size, out_size) in enumerate(zip(in_sizes, layers)):
            linears.append(
                (f"linear_{ix}", torch.nn.Linear(in_size, out_size, bias=True))
            )
            linears.append((f"norm_{ix}", torch.nn.BatchNorm1d(out_size)))
            linears.append((f"relu_{ix}", torch.nn.ReLU()))
        self.layers = torch.nn.Sequential(OrderedDict(linears))

        self.critic = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "critic_linear",
                        torch.nn.Linear(layers[-1], critic_hidden, bias=True),
                    ),
                    ("critic_norm", torch.nn.BatchNorm1d(critic_hidden)),
                    ("critic_relu", torch.nn.ReLU()),
                    (
                        "critic_output",
                        torch.nn.Linear(
                            critic_hidden,
                            1,
                            bias=False,
                        ),
                    ),
                ]
            )
        )
        self.action = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "action_linear",
                        torch.nn.Linear(layers[-1], action_hidden, bias=True),
                    ),
                    ("action_norm", torch.nn.BatchNorm1d(action_hidden)),
                    ("action_relu", torch.nn.ReLU()),
                    (
                        "action_output",
                        torch.nn.Linear(
                            action_hidden,
                            nb_actions,
                            bias=False,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward function for the simple network.

        Args:
            x: Input state to the network.

        Returns:
            Dict[str, torch.Tensor]: The output predictions of the network. These predictions
            include the critic value ('critic'), the softmaxed action distribution ('action'), and
            the non-softmaxed output for the action distribution ('rough_action').
        """
        inputs = [i.view(i.shape[0], -1) for i in x.values()]
        y = torch.cat(inputs, dim=-1)
        z = self.layers(y)
        value = self.critic(z)
        rough_action = self.action(z)
        return {"critic": value, "rough_action": rough_action}
