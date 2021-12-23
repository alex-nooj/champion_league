from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.scripted import SimpleHeuristic
from champion_league.utils.progress_bar import ProgressBar


class ImitationAgent(Agent):
    def __init__(
        self,
        device: int,
        network: nn.Module,
        lr: float,
        embed_battle: Callable[[Battle, Optional[bool]], Dict[str, Tensor]],
        logdir: str,
        tag: str,
    ):
        super().__init__(logdir, tag)
        self.device = device
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self._policy = SimpleHeuristic("simple_heuristic")
        self._embed_battle = embed_battle
        self._action_loss = torch.nn.CrossEntropyLoss()
        self._value_loss = torch.nn.MSELoss()

        keys = ["A. Acc", "A. Loss", "V. Acc", "V. Loss", "Total"]
        self.progress_bar = ProgressBar(keys)
        self.training_set = None
        self.validation_set = None

    def sample_action(
        self, state: Dict[str, Tensor], internals: Dict[str, Tensor]
    ) -> Tuple[float, float, float]:
        """Method for sampling an action from a distribution. This method should take in a
        state and return an action, log_probability, and the value of the current state.

        Parameters
        ----------
        state: Dict[str, torch.Tensor]
            The current, preprocessed state
        internals: Dict[str, torch.Tensor]
            The previous internals of the network.

        Returns
        -------
        Tuple[float, float, float]
            The action, log_probability, and value in that order
        """
        pass

    def act(self, state: Battle) -> int:
        """Uses the SimpleHeuristic agent to choose an action.

        Parameters
        ----------
        state: Battle
            The preprocessed state of the pokemon battle.

        Returns
        -------
        int
            The action chosen by the network.
        """
        return self._policy.act(state)

    def learn_step(self, data_loader: DataLoader, epoch: int) -> Dict[str, List[float]]:
        """Supervised learning step.

        Parameters
        ----------
        data_loader
        epoch: int

        Returns
        -------

        """
        self.network.train()

        epoch_stats = {
            "A. Acc": [],
            "A. Loss": [],
            "V. Acc": [],
            "V. Loss": [],
            "Total": [],
        }

        self.progress_bar.set_epoch(epoch)

        for i, (states, actions, values) in enumerate(data_loader):
            self.optimizer.zero_grad()
            batch_loss = self._predict_batch(states, actions, values)
            batch_loss["Total"].backward()
            self.optimizer.step()
            for key in epoch_stats:
                epoch_stats[key].append(torch.clone(batch_loss[key].detach()))
            self.progress_bar.print_bar(
                i / len(data_loader),
                {k: torch.stack(v).mean().item() for k, v in epoch_stats.items()},
                True,
            )

        return {k: torch.stack(v).mean().item() for k, v in epoch_stats.items()}

    @torch.no_grad()
    def validation_step(self, data_loader: DataLoader) -> Dict[str, List[float]]:
        """Supervised learning validation step.

        Parameters
        ----------
        data_loader

        Returns
        -------

        """
        self.network.eval()

        epoch_stats = {
            "A. Acc": [],
            "A. Loss": [],
            "V. Acc": [],
            "V. Loss": [],
            "Total": [],
        }

        for i, (states, actions, values) in enumerate(data_loader):
            batch_loss = self._predict_batch(states, actions, values)

            for key in epoch_stats:
                epoch_stats[key].append(torch.clone(batch_loss[key]))

            self.progress_bar.print_bar(
                i / len(data_loader),
                {k: torch.stack(v).mean().item() for k, v in epoch_stats.items()},
                False,
            )

        return {k: torch.stack(v).mean().item() for k, v in epoch_stats.items()}

    def embed_battle(self, battle: Battle, reset: Optional[bool]) -> Dict[str, Tensor]:
        return self._embed_battle(battle, reset)

    def _predict_batch(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        y, _ = self.network(x_internals={"x": states, "internals": {}})
        x_action = y["rough_action"]
        x_value = y["critic"]
        action_loss = self._action_loss(x_action, actions)

        value_loss = self._value_loss(x_value.squeeze(), values.squeeze())

        return {
            "A. Loss": action_loss,
            "V. Loss": value_loss,
            "Total": action_loss + value_loss,
            "A. Acc": torch.mean((torch.argmax(x_action, dim=-1) == actions).float()),
            "V. Acc": 1 - torch.mean(torch.abs(x_value - values)),
        }
