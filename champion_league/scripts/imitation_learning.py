import json
import os
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from champion_league.network import build_network_from_args
from champion_league.utils.directory_utils import check_and_make_dir
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.parse_args import parse_args
from champion_league.utils.progress_bar import ProgressBar


def save_args(args: DotDict):
    check_and_make_dir(args.logdir)
    check_and_make_dir(os.path.join(args.logdir, args.tag))

    with open(os.path.join(args.logdir, args.tag, "args.json"), "w") as fp:
        json.dump(args, fp, indent=2)


def save_model(save_path: str, network: torch.nn.Module):
    torch.save(network.state_dict(), save_path)


class PokeSet(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        device: int,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.device = f"cuda:{device}"

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx].float().to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].float().to(self.device),
        )


def create_datasets(
    dataset: Union[Dict[str, np.ndarray], NpzFile],
    split_ratio: float,
    device: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    states = torch.tensor(dataset["states"])
    actions = torch.tensor(dataset["actions"])
    rewards = torch.tensor(dataset["rewards"])

    split_idx = int(split_ratio * states.shape[0])

    train_set = PokeSet(
        states[:split_idx], actions[:split_idx], rewards[:split_idx], device
    )
    val_set = PokeSet(
        states[split_idx:], actions[split_idx:], rewards[split_idx:], device
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True),
    )


def eval_over_dataset(
    dataset: DataLoader,
    network: torch.nn.Module,
    progress_bar: ProgressBar,
    optim: Optional[Any] = None,
) -> Dict[str, float]:
    action_loss_fcn = torch.nn.CrossEntropyLoss()
    reward_loss_fcn = torch.nn.MSELoss()
    sample_count = 0
    action_acc = 0
    epoch_reward_loss = 0
    epoch_action_loss = 0
    epoch_total_loss = 0
    for batch_idx, (state, action, reward) in enumerate(dataset):
        preds, _ = network(state)
        action_acc += torch.sum(torch.argmax(preds["rough_action"], dim=-1) == action)

        sample_count += dataset.batch_size
        action_loss = action_loss_fcn(preds["rough_action"], action)
        reward_loss = reward_loss_fcn(preds["critic"].squeeze(-1), reward)
        total_loss = action_loss + reward_loss

        epoch_action_loss += action_loss
        epoch_reward_loss += reward_loss
        epoch_total_loss += total_loss

        if optim is not None:
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        progress_bar.print_bar(
            batch_idx / len(dataset),
            {
                "Accuracy": action_acc / sample_count,
                "Action Loss": epoch_action_loss / batch_idx,
                "Reward Loss": epoch_reward_loss / batch_idx,
                "Total Loss": epoch_total_loss / batch_idx,
            },
            optim is not None,
        )

    return {
        "Accuracy": action_acc / sample_count,
        "Action Loss": epoch_action_loss / len(dataset),
        "Reward Loss": epoch_reward_loss / len(dataset),
        "Total Loss": epoch_total_loss / len(dataset),
    }


def imitation_learning(
    dataset: Union[str, Dict[str, np.ndarray]],
    split_ratio: float,
    device: int,
    batch_size: int,
    nb_epochs: int,
    lr: float,
    network: torch.nn.Module,
    logdir: str,
    tag: str,
    patience: Optional[int],
) -> torch.nn.Module:

    writer = SummaryWriter(log_dir=os.path.join(logdir, tag))

    if type(dataset) == str:
        dataset = np.load(dataset)

    train_set, val_set = create_datasets(dataset, split_ratio, device, batch_size)

    check_and_make_dir(os.path.join(logdir, tag))
    check_and_make_dir(os.path.join(logdir, tag, "sl"))

    optim = torch.optim.Adam(network.parameters(), lr)

    progress_bar = ProgressBar(
        ["Accuracy", "Action Loss", "Reward Loss", "Total Loss"],
    )

    min_val_loss = None
    epochs_since_improvement = 0
    best_model = None
    for epoch in range(nb_epochs):
        if epoch != 0:
            progress_bar.set_epoch(epoch)
        training_results = eval_over_dataset(train_set, network, progress_bar, optim)

        for k, v in training_results.items():
            writer.add_scalar(f"Imitation Training/{k}", v, epoch)

        with torch.no_grad():
            validation_results = eval_over_dataset(val_set, network, progress_bar)
            for k, v in validation_results.items():
                writer.add_scalar(f"Imitation Validation/{k}", v, epoch)

            if min_val_loss is None or validation_results["Total Loss"] < min_val_loss:
                min_val_loss = validation_results["Total Loss"]
                best_model = OrderedDict(
                    [(k, torch.clone(v)) for k, v in network.state_dict().items()]
                )

                save_model(os.path.join(logdir, tag, "sl", "best_model.pt"), network)
                epochs_since_improvement = 0
            elif 0 < patience < epochs_since_improvement:
                break
            else:
                epochs_since_improvement += 1

    progress_bar.close()
    network.load_state_dict(best_model)
    return network


def main(args: DotDict) -> torch.nn.Module:
    dataset = np.load(args.dataset)

    args.in_shape = dataset["states"].shape[1:]

    network = build_network_from_args(args).to(f"cuda:{args.device}")

    return imitation_learning(
        dataset=dataset,
        split_ratio=args.split_ratio,
        device=args.device,
        batch_size=args.batch_size,
        nb_epochs=args.nb_epochs,
        lr=args.lr,
        network=network,
        logdir=os.path.join(args.logdir, "challengers"),
        tag=args.tag,
        patience=args.patience,
    )


if __name__ == "__main__":
    _ = main(parse_args())
