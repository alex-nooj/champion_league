import json
import os
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from champion_league.utils.directory_utils import check_and_make_dir
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir


class Agent:
    def __init__(self, logdir: str, tag: str, resume: Optional[bool] = False):
        """The base class for an agent. Implements some methods that are just useful for all agents
        or more useful as a standard (like tensorboard logging) while leaving the sample action
        method unimplemented.

        Parameters
        ----------
        logdir: str
            The path to the agent (should be up to and including "challengers" if this is a league
            agent
        tag: str
            The name of the agent
        resume: Optional[bool]
            Whether or not we're starting from a previously trained agent.
        """
        self.logdir = logdir
        self.tag = tag
        self.writer = SummaryWriter(log_dir=os.path.join(logdir, tag))
        self.index_dict = {}
        self.win_rates = {}

        if resume:
            try:
                self.reload_tboard(get_most_recent_epoch(os.path.join(logdir, tag)))
            except ValueError:
                pass

    def sample_action(
        self, state: Dict[str, torch.Tensor], internals: Dict[str, torch.Tensor]
    ) -> Tuple[float, float, float]:
        """Abstract method for sampling an action from a distribution. This method should take in a
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
        raise NotImplementedError

    def save_model(
        self, network: nn.Module, epoch: int, args: DotDict, title: Optional[str] = None
    ) -> None:
        """Saves the current network into an epoch directory so that it can be called back later.

        Parameters
        ----------
        network: nn.Module
            The network to be saved
        epoch: int
            The current epoch of training
        args: DotDict
            The arguments used to set up training
        title: Optional[str]
            An optional argument for naming the weights file

        Returns
        -------
        None
        """
        if title is None:
            title = f"network.pt"

        save_dir = self._get_save_dir(epoch)
        self._check_and_make_dir(save_dir)
        torch.save(network.state_dict(), os.path.join(save_dir, title))

        with open(os.path.join(save_dir, "args.json"), "w") as fp:
            json.dump(args, fp, indent=2)

        with open(os.path.join(save_dir, "tboard_info.json"), "w") as fp:
            json.dump(self.index_dict, fp, indent=2)

    def save_args(self, args: DotDict) -> None:
        """Saves just the arguments used to set up training

        Parameters
        ----------
        args: DotDict
            Arguments used to set up training

        Returns
        -------
        None
        """
        self._check_and_make_dir(self.logdir)
        self._check_and_make_dir(os.path.join(self.logdir, self.tag))

        with open(os.path.join(self.logdir, self.tag, "args.json"), "w") as fp:
            json.dump(args, fp, indent=2)

    def save_wins(self, epoch: int, win_rates: Dict[str, float]) -> None:
        """Saves the win rates of the current agent

        Parameters
        ----------
        epoch: int
            Which epoch of training the agent is on
        win_rates: Dict[str, float]
            The win rates against each agent in the league

        Returns
        -------
        None
        """
        save_dir = self._get_save_dir(epoch)
        self._check_and_make_dir(save_dir)

        with open(os.path.join(save_dir, "win_rates.json"), "w") as fp:
            json.dump(win_rates, fp, indent=2)

    def _get_save_dir(self, epoch: int) -> str:
        """Helper function for getting the name of the epoch directory.

        Parameters
        ----------
        epoch: int
            The desired epoch

        Returns
        -------
        str
            The full path with the proper naming convention.
        """
        return get_save_dir(self.logdir, self.tag, epoch)

    @staticmethod
    def _check_and_make_dir(path: str) -> None:
        """First checks if a directory exists then creates it if it doesn't.

        Parameters
        ----------
        path: str
            The path of the directory to be created

        Returns
        -------
        None
        """
        check_and_make_dir(path)

    def write_to_tboard(self, label: str, value: Union[float, np.ndarray]) -> None:
        """Writes a value to tensorboard, while keeping track of the tensorboard index.

        Parameters
        ----------
        label: str
            The label for the value to display on tensorboard
        value: Union[float, npt.NDArray]
            The value to add to tensorboard

        Returns
        -------
        None
        """
        if label not in self.index_dict:
            self.index_dict[label] = 0

        self.writer.add_scalar(label, value, self.index_dict[label])
        self.index_dict[label] += 1

    def reload_tboard(self, epoch: int) -> None:
        """Function for resuming a previously training agent. This will load all of the tensorboard
        categories as well as the latest index. If the file is not found, then we resume from the
        start.

        Parameters
        ----------
        epoch: int
            Where to look for the tboard_info.json file

        Returns
        -------
        None
        """
        try:
            with open(
                os.path.join(self._get_save_dir(epoch), "tboard_info.json"), "r"
            ) as fp:
                self.index_dict = json.load(fp)
        except FileNotFoundError:
            pass

    def update_winrates(self, opponent: str, win: int) -> None:
        """Function for tracking the win-rates of the agent with a sliding window.

        Parameters
        ----------
        opponent: str
            The name of the opponent (no path)
        win: int
            1 if the agent won, 0 if the agent lost

        Returns
        -------
        None
        """
        if opponent not in self.win_rates:
            self.win_rates[opponent] = [win]
        elif len(self.win_rates[opponent]) > 100:
            self.win_rates[opponent] = self.win_rates[opponent][-1:] + [win]
        else:
            self.win_rates[opponent].append(win)
