import json
import os
from typing import Optional, Dict

import torch
from adept.utils.util import DotDict
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class BaseAgent:
    def __init__(self, logdir: str, tag: str):
        self.logdir = logdir
        self.tag = tag
        self.writer = SummaryWriter(log_dir=os.path.join(logdir, tag))

    def sample_action(self, state: torch.Tensor):
        raise NotImplementedError

    def save_model(
        self, network: nn.Module, epoch: int, args: DotDict, title: Optional[str] = None
    ):
        if title is None:
            title = f"network.pt"

        save_dir = self._get_save_dir(epoch)
        self._check_and_make_dir(save_dir)
        torch.save(network.state_dict(), os.path.join(save_dir, title))

        with open(os.path.join(save_dir, "args.json"), "w") as fp:
            json.dump(args, fp, indent=2)

    def save_args(self, args: DotDict):
        self._check_and_make_dir(self.logdir)
        self._check_and_make_dir(os.path.join(self.logdir, self.tag))

        with open(os.path.join(self.logdir, self.tag, "args.json"), "w") as fp:
            json.dump(args, fp, indent=2)

    def save_wins(self, epoch: int, win_rates: Dict[str, float]):
        save_dir = self._get_save_dir(epoch)
        self._check_and_make_dir(save_dir)

        with open(os.path.join(save_dir, "win_rates.json"), "w") as fp:
            json.dump(win_rates, fp, indent=2)

    def _get_save_dir(self, epoch: int) -> str:
        return os.path.join(self.logdir, self.tag, f"{self.tag}_{epoch:05d}")

    @staticmethod
    def _check_and_make_dir(path: str):
        if not os.path.isdir(path):
            os.mkdir(path)

    def write_to_tboard(self, label: str, value: float, idx: int):
        self.writer.add_scalar(label, value, idx)
