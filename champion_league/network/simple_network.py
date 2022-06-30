from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from champion_league.config.load_configs import get_default_args
from champion_league.network.base_network import BaseNetwork
from champion_league.network.modules.linear import LinearMod


class SimpleNetwork(BaseNetwork):
    def __init__(
        self,
        nb_actions: int,
        in_shape: Dict[str, Tuple[int]],
        nb_layers: Optional[int] = None,
        nb_hidden: Optional[int] = None,
        critic_layers: Optional[int] = None,
        critic_hidden: Optional[int] = None,
        action_layers: Optional[int] = None,
        action_hidden: Optional[int] = None,
    ):
        super().__init__()
        cfg = get_default_args(__file__)
        nb_layers = nb_layers or cfg.nb_layers
        nb_hidden = nb_hidden or cfg.nb_hidden
        critic_layers = critic_layers or cfg.critic_layers
        critic_hidden = critic_hidden or cfg.critic_hidden
        action_layers = action_layers or cfg.action_layers
        action_hidden = action_hidden or cfg.action_hidden

        total_in_shape = sum([v[0] for v in in_shape.values()])
        self.layers = LinearMod(
            (total_in_shape,), nb_hidden=nb_hidden, nb_layers=nb_layers
        )

        self.critic = LinearMod(
            self.layers.output_shape,
            nb_hidden=critic_hidden,
            nb_layers=critic_layers,
            output_size=1,
        )
        self.action = LinearMod(
            self.layers.output_shape,
            nb_hidden=action_hidden,
            nb_layers=action_layers,
            output_size=nb_actions,
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward function for the simple network.

        Args:
            x: Input state to the network.

        Returns:
            Dict[str, torch.Tensor]: The output predictions of the network. These predictions
            include the critic value ('critic'), the softmaxed action distribution ('action'), and
            the non-softmaxed output for the action distribution ('rough_action').
        """
        inputs = [i for i in x.values()]
        y = torch.cat(inputs, dim=-1)
        z = self.layers(y)
        value = self.critic(z)
        rough_action = self.action(z)
        soft_action = self.softmax_layer(rough_action)
        return {"action": soft_action, "critic": value, "rough_action": rough_action}
