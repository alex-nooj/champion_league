import json
import os
from abc import abstractmethod
from typing import Union, Optional, Dict, List

import torch
from adept.utils.util import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle

# Should include:
# def _action_to_move
from torch.utils.tensorboard import SummaryWriter


class Agent:
    _ACTION_SPACE = None
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001

    def __init__(
        self, args: DotDict,
    ):
        self._args = args
        self.win_rates = {}
        self.save_args(0)

        self._summary_writer = SummaryWriter(os.path.join(args.logdir, args.agent_type, args.tag))

    def save_args(self, epoch):
        if not os.path.isdir(self._args.logdir):
            os.mkdir(self._args.logdir)

        if not os.path.isdir(os.path.join(self._args.logdir, "challengers", self._args.tag)):
            os.mkdir(os.path.join(self._args.logdir, "challengers", self._args.tag))

        savedir = os.path.join(
            self._args.logdir, "challengers", self._args.tag, f"{self._args.tag}_{epoch}"
        )

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        with open(os.path.join(savedir, "args.json"), "w") as fp:
            json.dump(self._args, fp)

    def save_model(self, network, epoch):
        savedir = os.path.join(
            self._args.logdir, "challengers", self._args.tag, f"{self._args.tag}_{epoch}"
        )

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        torch.save(network.state_dict(), os.path.join(savedir, f"{self._args.tag}_{epoch}.pt"))

    def save_wins(self, epoch, win_rates):
        savedir = os.path.join(
            self._args.logdir, "challengers", self._args.tag, f"{self._args.tag}_{epoch}"
        )

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        with open(os.path.join(savedir, "win_rates.json"), "w") as fp:
            json.dump(win_rates, fp)

    def load_model(self, network):
        directory_contents = os.listdir(os.path.join(self._args.logdir, self._args.tag))

        epoch_directories = [
            int(directory_name.rsplit("_")[-1])
            for directory_name in directory_contents
            if "epoch" in directory_name
        ]

        epoch_directories.sort()
        epoch_directories.reverse()
        for epoch in epoch_directories:
            try:
                checkpoint = torch.load(
                    os.path.join(self._args.logdir, self._args.tag, f"{self._args.tag}_{epoch}",),
                    map_location=lambda storage, loc: storage,
                )
                network.load_state_dict(checkpoint)

            except:
                continue
            finally:
                break
        return network

    @abstractmethod
    def learn_step(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, battle: Union[AbstractBattle, Battle]):
        raise NotImplementedError

    def log_to_tensorboard(
        self,
        nb_steps: int,
        loss: Optional[float] = None,
        win_rates: Optional[Dict[str, List[int]]] = None,
        reward: Optional[int] = None,
    ) -> None:
        """Function for logging agent success to tensorboard

        Parameters
        ----------
        nb_steps: int
            How many training steps have been performed
        loss: Optional[float]
            The current, total loss of the agent
        win_rates: Optional[Dict[str, List[int]]]
            The win rates of the agent against all of the other agents
        reward: Optional[int]
            The end reward of each game

        Returns
        -------
        None
        """

        if loss is not None:
            self._summary_writer.add_scalar(
                "Loss/total_loss", loss, nb_steps
            )

        if win_rates is not None:
            total_wins = 0
            total_games = 0
            for key in win_rates:
                self._summary_writer.add_scalar(
                    f"Win_Rates/{key}", win_rates[key][0]/win_rates[key][1], nb_steps
                )
                total_wins += win_rates[key][0]
                total_games += win_rates[key][1]
            self._summary_writer.add_scalar(
                "Win_Rates/total", total_wins / total_games, nb_steps
            )

        if reward is not None:
            self._summary_writer.add_scalar(
                f"Reward/reward", reward, nb_steps
            )
