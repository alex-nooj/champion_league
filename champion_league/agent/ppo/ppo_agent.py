from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from champion_league.agent.base.base_agent import BaseAgent


def ac_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_clip: Optional[float] = 0.2,
):
    probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_probabiliy_ratios = torch.clamp(probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip)

    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probabiliy_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        device: float,
        network: torch.nn.Module,
        lr: float,
        entropy_weight: float,
        clip: float,
        logdir: str,
        tag: str,
        mini_epochs: Optional[int] = 4,
    ):
        super().__init__(logdir, tag)
        self.device = device
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.entropy_weight = entropy_weight
        self.clip = clip

        self.mini_epochs = mini_epochs
        self.updates = 0

    def sample_action(self, state: torch.Tensor) -> Tuple[int, int, int]:
        if len(state.size()) == 2:
            state = state.unsqueeze(0)

        y = self.network(state)

        dist = Categorical(y["action"])

        action = dist.sample()

        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item(), y["critic"].item()

    def act(self, state: torch.Tensor) -> int:
        if len(state.size()) == 2:
            state = state.unsqueeze(0)

        y = self.network(state)

        return torch.argmax(y["action"], -1).item()

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.network(states.squeeze(1))

        dist = Categorical(y["action"])
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy, y["critic"].squeeze(1)

    def learn_step(self, data_loader: DataLoader) -> Dict[str, List[float]]:
        self.network = self.network.train()
        self.updates += 1
        epoch_losses = {"Policy Loss": [], "Entropy Loss": [], "Value Loss": [], "Total Loss": []}
        for i in range(self.mini_epochs):
            losses = {"Policy Loss": [], "Entropy Loss": [], "Value Loss": [], "Total Loss": []}
            for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
                actions = actions.long().to(self.device)
                advantages = advantages.float().to(self.device)
                old_log_probabilities = log_probabilities.float().to(self.device)

                self.optimizer.zero_grad()

                new_log_probabilities, entropy, values = self.evaluate_actions(
                    observations, actions
                )

                policy_loss = ac_loss(
                    new_log_probabilities, old_log_probabilities, advantages, epsilon_clip=self.clip
                ).mean()

                entropy_loss = self.entropy_weight * entropy.mean()

                value_loss = (
                    0.5 * (values.squeeze(-1) - rewards_to_go.to(values.device)).pow(2).mean()
                )

                loss = policy_loss + entropy_loss + value_loss

                loss.backward()

                self.optimizer.step()

                losses["Policy Loss"].append(policy_loss.item())
                losses["Entropy Loss"].append(entropy_loss.item())
                losses["Value Loss"].append(value_loss.item())
                losses["Total Loss"].append(loss.item())
            for key in epoch_losses:
                epoch_losses[key].append(np.mean(losses[key]))

        self.network = self.network.eval()
        return epoch_losses
