import json
import os
from abc import abstractmethod
from typing import Union

import torch
from adept.utils.util import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle

# Should include:
# def _action_to_move
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
        savedir = os.path.join(self._args.logdir, "challengers", self._args.tag, f"{self._args.tag}_{epoch}")

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        torch.save(
            network.state_dict(), os.path.join(savedir, f"{self._args.tag}_{epoch}.pt")
        )

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
                    os.path.join(
                        self._args.logdir,
                        self._args.tag,
                        f"{self._args.tag}_{epoch}",
                    ),
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
