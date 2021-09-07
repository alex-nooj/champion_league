from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from champion_league.agent.base.base_agent import Agent


def ac_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    epsilon_clip: Optional[float] = 0.2,
) -> torch.Tensor:
    """Calulates the policy loss for PPO.

    Parameters
    ----------
    new_log_probabilities: torch.Tensor
        The log probabilites that have just been output by the network.
    old_log_probabilities: torch.Tensor
        The log probabilities that were originally output by the network.
    advantages: torch.Tensor
        The advantages, estimated by the network.
    epsilon_clip: Optional[float]
        The clipping value used by PPO to restrict how much the network can update.

    Returns
    -------
    torch.Tensor
        The policy loss, as defined in PPO.
    """
    probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_probabiliy_ratios = torch.clamp(
        probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip
    )

    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probabiliy_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)


class PPOAgent(Agent):
    def __init__(
        self,
        device: int,
        network: nn.Module,
        lr: float,
        entropy_weight: float,
        clip: float,
        logdir: str,
        tag: str,
        mini_epochs: Optional[int] = 4,
        resume: Optional[bool] = False,
    ):
        """This is the agent used to train a network using PPO. This class handles the sampling of
        moves, the acting when not sampling, and the learning. Built on the base Agent class.

        Parameters
        ----------
        device: int
            Which device the network is on.
        network: nn.Module
            The network being trained.
        lr: float
            The learning rate for the network.
        entropy_weight: float
            How much to weigh the entropy loss.
        clip: float
            The clip value (epsilon) used in PPO to clip the loss.
        logdir: str
            The path to the network's directory (up to "challengers")
        tag: str
            The name of the agent.
        mini_epochs: Optional[int]
            How many mini-epochs to run during the update.
        resume: Optional[bool]
            Whether or not we're starting from a previously trained agent.
        """
        super().__init__(logdir, tag, resume)
        self.device = device
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.entropy_weight = entropy_weight
        self.clip = clip

        self.mini_epochs = mini_epochs
        self.updates = 0

    def sample_action(self, state: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        """Samples an action from a distribution using the network that's training.

        Parameters
        ----------
        state: torch.Tensor
            The current, preprocessed state

        Returns
        -------
        Tuple[float, float, float]
            The action, log_probability, and value in that order
        """
        y = self.network(state)

        dist = Categorical(y["action"])

        action = dist.sample()

        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item(), y["critic"].item()

    def act(self, state: torch.Tensor) -> int:
        """Rather than sample the output distribution for an action, this function takes the acton
        the network believes is best.

        Parameters
        ----------
        state: torch.Tensor
            The preprocessed state of the pokemon battle.

        Returns
        -------
        int
            The action chosen by the network.
        """
        y = self.network(state)

        return torch.argmax(y["action"], -1).item()

    def evaluate_actions(
        self, states: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This function calculates the log probabilities and entropy of the current distribution,
        as well as the critic's output.

        Parameters
        ----------
        states: Dict[str, torch.Tensor]
            The preprocessed state of the pokemon battle.
        actions: torch.Tensor
            The actions that were selected by the agent when gathering the rollout.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The log probs, entropy, and critic values.
        """
        y = self.network(states)

        dist = Categorical(y["action"])
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy, y["critic"].squeeze(1)

    def learn_step(self, data_loader: DataLoader) -> Dict[str, List[float]]:
        """The PPO learn step. This updates the network using the given rollout.

        Parameters
        ----------
        data_loader: DataLoader
            The rollout to train on.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary with keys 'Policy Loss', 'Entropy Loss', 'Value Loss', and 'Total Loss' and
            values representing those losses for each epoch.
        """
        self.network = self.network.train()
        self.updates += 1
        epoch_losses = {
            "Policy Loss": [],
            "Entropy Loss": [],
            "Value Loss": [],
            "Total Loss": [],
        }

        for i in tqdm(range(self.mini_epochs)):
            losses = {
                "Policy Loss": [],
                "Entropy Loss": [],
                "Value Loss": [],
                "Total Loss": [],
            }
            for (
                observations,
                actions,
                advantages,
                log_probabilities,
                rewards_to_go,
            ) in data_loader:
                actions = actions.long().to(self.device)
                advantages = advantages.float().to(self.device)
                observations = {k: v.squeeze(1) for k, v in observations.items()}
                old_log_probabilities = log_probabilities.float().to(self.device)

                self.optimizer.zero_grad()

                new_log_probabilities, entropy, values = self.evaluate_actions(
                    observations, actions
                )

                policy_loss = ac_loss(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=self.clip,
                ).mean()

                entropy_loss = self.entropy_weight * entropy.mean()

                value_loss = (
                    0.5
                    * (values.squeeze(-1) - rewards_to_go.to(values.device))
                    .pow(2)
                    .mean()
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
