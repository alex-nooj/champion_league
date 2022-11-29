from collections import deque
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from omegaconf import OmegaConf
from poke_env.teambuilder.teambuilder import Teambuilder
from torch import nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from champion_league.agent.ppo.ppo_replay_buffer import PPOReplayBuffer
from champion_league.preprocessor import Preprocessor
from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.directory_utils import PokePath


class Agent:
    def __init__(
        self,
        league_path: PokePath,
        tag: str,
        network: nn.Module,
        device: int,
        replay_buffer: PPOReplayBuffer,
        resume: Optional[bool] = False,
    ):
        """The base class for an agent. Implements some methods that are just useful for all agents
        or more useful as a standard (like tensorboard logging) while leaving the sample action
        method unimplemented.

        Args:
            league_path: PokePath for this league.
            tag: The name of the agent
            resume: Whether we're starting from a previously trained agent.
        """
        self.league_path = league_path
        self.logdir = league_path
        self.tag = tag
        self.network = network
        self.device = device
        self.replay_buffer = replay_buffer
        self.writer = SummaryWriter(log_dir=str(league_path.agent))
        self.index_dict = {}
        self.win_rates = {}

        if resume:
            try:
                self.reload_tboard(get_most_recent_epoch(league_path.agent))
            except ValueError:
                pass

    def sample_action(self, state: Dict[str, Tensor]) -> Tuple[int, float, float]:
        """Abstract method for sampling an action from a distribution.

        This method should take in a state and return an action, log_probability, and the value of
        the current state.

        Args:
            state: The current, preprocessed state

        Returns:
            Tuple[int, float, float]: The action, log_probability, and value in that order
        """
        raise NotImplementedError

    def act(self, state: Dict[str, Tensor]) -> int:
        """Abstract method for taking the action the agent believes is best.

        Args:
            state: The preprocessed state of the Pokemon battle.

        Returns:
            Dict[str, torch.Tensor]: The action chosen by the network.
        """
        raise NotImplementedError

    def learn_step(self, epoch: int):
        """Function for training the network."""
        raise NotImplementedError

    def save_model(
        self,
        epoch: int,
        network: nn.Module,
        preprocessor: Preprocessor,
        team_builder: Teambuilder,
    ) -> None:
        """Saves the current network into an epoch directory so that it can be called back later.

        Args:
            epoch: The current epoch of training
            network: The network to be saved
            preprocessor: The preprocessor that the agent uses
            team_builder: The agent's team builder.
        """
        save_dir = get_save_dir(self.league_path.agent, epoch)
        save_file = save_dir / "network.pth"
        torch.save(
            {
                "network": network,
                "preprocessor": preprocessor,
                "team": team_builder,
            },
            str(save_file),
        )
        OmegaConf.save(config=self.index_dict, f=str(save_dir / "tboard_info.yaml"))

    def save_wins(self, epoch: int, win_rates: Dict[str, float]) -> None:
        """Saves the win rates of the current agent

        Args:
            epoch: Which epoch of training the agent is on
            win_rates: The win rates against each agent in the league
        """
        save_dir = get_save_dir(self.league_path.agent, epoch)

        OmegaConf.save(config=win_rates, f=save_dir / "win_rates.yaml")

    def log_scalar(self, label: str, value: Union[float, np.ndarray]) -> None:
        """Writes a value to tensorboard, while keeping track of the tensorboard index.

        Args:
            label: The label for the value to display on tensorboard
            value: The value to add to tensorboard
        """
        if label not in self.index_dict:
            self.index_dict[label] = 0

        self.writer.add_scalar(label, value, self.index_dict[label])
        self.index_dict[label] += 1

    def reload_tboard(self, epoch: int) -> None:
        """Function for resuming a previously training agent. This will load all of the tensorboard
        categories as well as the latest index. If the file is not found, then we resume from the
        start.

        Args:
            epoch: Where to look for the tboard_info.json file
        """
        try:
            self.index_dict = OmegaConf.to_container(
                OmegaConf.load(
                    get_save_dir(self.league_path.agent, epoch) / "tboard_info.yaml"
                )
            )
        except FileNotFoundError:
            pass

    def update_winrates(self, opponent: str, win: int) -> None:
        """Function for tracking the win-rates of the agent with a sliding window.

        Args:
            opponent: The name of the opponent (no path)
            win: 1 if the agent won, 0 if the agent lost
        """
        if opponent not in self.win_rates:
            self.win_rates[opponent] = deque([0 for _ in range(100)])

        self.win_rates[opponent].append(win)
        self.win_rates[opponent].popleft()
