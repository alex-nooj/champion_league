import sys
from typing import Tuple
from typing import List
from typing import Dict
from typing import Optional

import numpy as np
import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from champion_league.agent.base.base_agent import BaseAgent


def centered(in_str: str, desired_len: int) -> str:
    return in_str.center(max(len(in_str), desired_len))


class ImitationAgent(BaseAgent):
    def __init__(
        self,
        device: float,
        network: torch.nn.Module,
        lr: float,
        logdir: str,
        tag: str,
        mini_epochs: Optional[int] = 10,
        max_bar_length: Optional[int] = 40,
    ):
        super().__init__(logdir, tag)
        self.device = device
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.loss = torch.nn.CrossEntropyLoss()
        self.mini_epochs = mini_epochs
        self._max_bar_length = max_bar_length
        self._stdout = None
        self.epoch = 0

    def learn_step(self, data_loader: DataLoader) -> Dict[str, float]:
        self.network = self.network.train()
        epoch_losses = {"Accuracy": [], "Action Loss": [], "Value Loss": [], "Total Loss": []}
        for observations, actions, _, _, rewards_to_go in data_loader:
            actions = actions.long().to(self.device)

            self.optimizer.zero_grad()

            y = self.network(observations)
            action_loss = self.loss(y["rough_action"], actions)
            value_loss = (
                0.5 * (y["critic"].squeeze(-1) - rewards_to_go.to(y["critic"].device)).pow(2).mean()
            )

            loss = action_loss + value_loss

            loss.backward()
            self.optimizer.step()
            epoch_losses["Accuracy"].append(
                torch.mean((torch.argmax(y["action"], dim=-1) == actions).float()).item()
            )
            epoch_losses["Action Loss"].append(action_loss.item())
            epoch_losses["Value Loss"].append(value_loss.item())
            epoch_losses["Total Loss"].append(loss.item())
        for key in epoch_losses:
            epoch_losses[key] = np.mean(epoch_losses[key])

        return epoch_losses

    @torch.no_grad()
    def validation_step(self, data_loader: DataLoader):
        validation_losses = {"Accuracy": [], "Action Loss": [], "Value Loss": [], "Total Loss": []}
        for observations, actions, _, _, rewards_to_go in data_loader:
            actions = actions.long().to(self.device)
            y = self.network(observations)
            action_loss = self.loss(y["rough_action"], actions)
            value_loss = (
                0.5 * (y["critic"].squeeze(-1) - rewards_to_go.to(y["critic"].device)).pow(2).mean()
            )
            total_loss = action_loss + value_loss
            validation_losses["Accuracy"].append(
                torch.mean((torch.argmax(y["action"], dim=-1) == actions).float()).item()
            )
            validation_losses["Action Loss"].append(action_loss)
            validation_losses["Value Loss"].append(value_loss)
            validation_losses["Total Loss"].append(total_loss)

        return {k: np.mean(v) for k, v in validation_losses.items()}

    def sample_action(self, state: torch.Tensor) -> int:
        if len(state.size()) == 2:
            state.unsqueeze(0)

        y = self.network(state)
        return torch.argmax(y["action"], -1).item()

    def progress_bar(
        self,
        loss: Dict[str, float],
        batches_completed: int,
        batches_total: int,
        epoch: int,
        train_step: bool,
    ):
        # Determine how long the progress bar should be
        bar_length = round(self._max_bar_length * (batches_completed / batches_total)) + 1

        # Ensure its not longer than the maximum length
        bar_length = min((self._max_bar_length, bar_length))

        bar_char = "-" if train_step else "="
        pointer_char = ">" if train_step else "="
        space_char = " " if train_step else "-"

        # Build the progress bar
        bar = (
            "["
            + bar_char * (bar_length - 1)
            + pointer_char
            + space_char * (self._max_bar_length - np.max((bar_length, 1)))
            + "]"
        )

        epoch_section = centered("Epoch", 0)
        bar_section = " " * self._max_bar_length
        t_acc_section = centered("T. Acc", 0)
        t_action_loss_section = centered("T. Action Loss", 0)
        t_value_loss_section = centered("T. Value Loss", 0)
        t_total_loss_section = centered("T. Total Loss", 0)

        v_acc_section = centered("V. Acc", 0)
        v_action_loss_section = centered("V. Action Loss", 0)
        v_value_loss_section = centered("V. Value Loss", 0)
        v_total_loss_section = centered("V. Total Loss", 0)

        divider = (
            "+"
            + "-" * len(epoch_section)
            + "+"
            + "-" * len(bar_section)
            + "+"
            + "-" * len(t_acc_section)
            + "+"
            + "-" * len(t_action_loss_section)
            + "+"
            + "-" * len(t_value_loss_section)
            + "+"
            + "-" * len(t_total_loss_section)
            + "+"
            + "-" * len(v_acc_section)
            + "+"
            + "-" * len(v_action_loss_section)
            + "+"
            + "-" * len(v_value_loss_section)
            + "+"
            + "-" * len(v_total_loss_section)
            + "+"
        )

        title_bar = (
            "|"
            + epoch_section
            + "|"
            + bar_section
            + "|"
            + t_acc_section
            + "|"
            + t_action_loss_section
            + "|"
            + t_value_loss_section
            + "|"
            + t_total_loss_section
            + "|"
            + v_acc_section
            + "|"
            + v_action_loss_section
            + "|"
            + v_value_loss_section
            + "|"
            + v_total_loss_section
            + "|"
        )
        if self._stdout is None:
            print(divider)
            print(title_bar)
            print(divider)

        # If it's the training step, we should update self._stdout (this is used to track the
        # accuracies and loss for the training step only to be easily printed out)
        if train_step:
            self._stdout = (
                centered(f"{loss['Accuracy']}:0.3f", len(t_acc_section))
                + "|"
                + centered(f"{loss['Action Loss']}:0.3f", len(t_action_loss_section))
                + "|"
                + centered(f"{loss['Value Loss']}:0.3f", len(t_value_loss_section))
                + "|"
                + centered(f"{loss['Total Loss']}:0.3f", len(t_total_loss_section))
            )
            val_section = (
                "|"
                + centered(" ", len(v_acc_section))
                + "|"
                + centered(" ", len(v_action_loss_section))
                + "|"
                + centered(" ", len(v_value_loss_section))
                + "|"
                + centered(" ", len(v_total_loss_section))
            )
        else:
            val_section = (
                centered(f"{loss['Accuracy']}:0.3f", len(v_acc_section))
                + "|"
                + centered(f"{loss['Action Loss']}:0.3f", len(v_action_loss_section))
                + "|"
                + centered(f"{loss['Value Loss']}:0.3f", len(v_value_loss_section))
                + "|"
                + centered(f"{loss['Total Loss']}:0.3f", len(v_total_loss_section))
            )

        output = "\r|" + centered(str(epoch), len(epoch_section)) + f"|{bar}|" + self._stdout + "|"
        sys.stdout.write(output + self._stdout + val_section)
        sys.stdout.flush()
