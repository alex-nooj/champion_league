import copy

import json
import os
import torch
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from torch import nn
from typing import Any
from typing import Dict
from typing import Optional

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.network import build_network_from_args
from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.directory_utils import DotDict


class LeaguePlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        device: int,
        network: nn.Module,
        preprocessor: Preprocessor,
        sample_moves: Optional[bool] = True,
        **kwargs: Dict[str, Any],
    ):
        """This is the player for the league. It acts as the opponent for the training agent and
        handles all server communications.

        Parameters
        ----------
        device: int
            The device to load the networks onto.
        network: nn.Module
            The network that is currently training. It will be periodically used for self-play.
        preprocessor: Preprocessor
            The preprocessor that the agent is using.
        sample_moves: Optional[bool]
            Whether to sample the output distribution of the network.
        kwargs: Dict[str, Any]
            Additional keyword arguments. Any of those used by PokeEnv would be placed here, such as
            player configurations, server configurations, or player avatar.
        """
        super().__init__(**kwargs)
        self.sample_moves = sample_moves

        self.opponent = None
        self.mode = None
        self.device = f"cuda:{device}"
        self.network = None
        self.update_network(network)
        self.preprocessor = preprocessor

        self.tag = self.change_agent("self")

    def choose_move(self, battle: Battle) -> BattleOrder:
        """The function used to pass the current state into the network or scripted agent and
        receive a battle order.

        Parameters
        ----------
        battle: Battle
            The current state in its raw form.

        Raises
        ------
        RuntimeError
            If an agent has not been selected yet and the server asks for a move, this error is
            raised.

        Returns
        -------
        BattleOrder
            The move that the agent would like to select, converted into a form that is readable by
            PokeEnv and Showdown.
        """
        if self.mode is None:
            raise RuntimeError("Agent cannot be none!")

        return self.opponent.choose_move(battle)

    def change_agent(
        self,
        agent_path: str,
    ) -> str:
        """This handles the swapping of the agent choosing the moves. For the league, this is useful
        as it allows the agent to see many different selection strategies during training, making it
        more robust. From a software engineering standpoint, this also allows us to switch agents
        really quickly while maintaining the current connection to the server.

        NOTE: LeaguePlayer does not handle the logic of which agent to select, it just switches to
        the desired agent.

        Parameters
        ----------
        agent_path: str
            The path to the chosen agent.

        Returns
        -------
        str
            The name of the current agent, or 'self' if we are currently running self-play
        """

        if agent_path == "self":
            # If we're doing self-play, then this loads up an opponent that is a copy of the current
            # network.
            self.opponent = RLOpponent(
                network=self.network,
                preprocessor=self.preprocessor,
                device=self.device,
                sample_moves=self.sample_moves,
            )
            self.mode = "self"
            self.tag = "self"
        else:
            # Otherwise, we're playing a league agent, so we have to build that network. So first we
            # load up the arguments, which will act as build instructions.
            with open(os.path.join(agent_path, "args.json"), "r") as fp:
                args = json.load(fp)
                args = DotDict(args)

            if "scripted" in args:
                # Scripted agents act differently than ML agents, so we have to treat them a little
                # differently.
                self.mode = "scripted"
                self.opponent = SCRIPTED_AGENTS[args.agent]
            else:
                # Otherwise, we have an ML agent, and have to build the LeagueOpponent class using
                # this network as a selection strategy.
                self.mode = "ml"
                args.resume = False
                network = build_network_from_args(args).eval()
                network.load_state_dict(
                    torch.load(os.path.join(agent_path, "network.pt"))
                )
                preprocessor = build_preprocessor_from_args(args)
                self.opponent = RLOpponent(
                    network=network,
                    preprocessor=preprocessor,
                    device=self.device,
                    sample_moves=self.sample_moves,
                )

            # Sometimes, we aren't actually fighting a league opponent and are instead just fighting
            # an earlier iteration of the current agent, which should still count as self-play
            if agent_path.rsplit("/")[-3] == "challengers":
                self.tag = "self"
            else:
                self.tag = agent_path.rsplit("/")[-1]

        return self.tag

    def update_network(self, network: nn.Module) -> None:
        """Performs a deep copy of the network. Useful for handling self-play.

        Parameters
        ----------
        network: nn.Module
            The network that should replace the current one stored by the agent.

        Returns
        -------
        None
        """
        self.network = copy.deepcopy(network).eval()
