import copy
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import torch
from omegaconf import OmegaConf
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from torch import nn

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.network import NETWORKS
from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors import PREPROCESSORS
from champion_league.teams.team_builder import AgentTeamBuilder


class LeaguePlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        device: int,
        network: nn.Module,
        preprocessor: Preprocessor,
        sample_moves: Optional[bool] = True,
        team: Optional[AgentTeamBuilder] = None,
        training_team: Optional[AgentTeamBuilder] = None,
        **kwargs: Any,
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
        kwargs: Any
            Additional keyword arguments. Any of those used by PokeEnv would be placed here, such as
            player configurations, server configurations, or player avatar.
        """
        super().__init__(team=team, **kwargs)
        self.sample_moves = sample_moves

        self.opponent = None
        self.mode = None
        self.device = device
        self.network = None
        self.update_network(network)
        self.preprocessor = preprocessor
        self.training_team = training_team
        self.team = team
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
        agent_path: Union[str, Path],
    ) -> str:
        """This handles the swapping of the agent choosing the moves. For the league, this is useful
        as it allows the agent to see many different selection strategies during training, making it
        more robust. From a software engineering standpoint, this also allows us to switch agents
        really quickly while maintaining the current connection to the server.

        NOTE: LeaguePlayer does not handle the logic of which agent to select, it just switches to
        the desired agent.

        Parameters
        ----------
        agent_path: Union[str, Path]
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
            self._team = self.training_team
        else:
            # Otherwise, we're playing a league agent, so we have to build that network. So first we
            # load up the arguments, which will act as build instructions.

            args = OmegaConf.to_container(OmegaConf.load(Path(agent_path, "args.yaml")))
            self._team = self.team
            if "scripted" in args:
                # Scripted agents act differently than ML agents, so we have to treat them a little
                # differently.
                self.mode = "scripted"
                self.opponent = SCRIPTED_AGENTS[args["agent"]]
                self._team.clear_team()
            else:
                # Otherwise, we have an ML agent, and have to build the LeagueOpponent class using
                # this network as a selection strategy.
                self.mode = "ml"
                preprocessor = PREPROCESSORS[args["preprocessor"]](
                    args["device"], **args[args["preprocessor"]]
                )
                network = NETWORKS[args["network"]](
                    nb_actions=args["nb_actions"],
                    in_shape=preprocessor.output_shape,
                    **args[args["network"]],
                ).eval()
                network.load_state_dict(torch.load(Path(agent_path, "network.pt")))

                self.opponent = RLOpponent(
                    network=network,
                    preprocessor=preprocessor,
                    device=self.device,
                    sample_moves=self.sample_moves,
                )
                self._team.load_new_team(agent_path)

            # Sometimes, we aren't actually fighting a league opponent and are instead just fighting
            # an earlier iteration of the current agent, which should still count as self-play
            if list(agent_path.parents)[2] == "challengers":
                self.tag = "self"
            else:
                self.tag = agent_path.stem

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

    def reset(self):
        if isinstance(self.opponent, RLOpponent):
            self.opponent.reset()

    def teampreview(self, battle):
        return "/team " + "".join([str(i + 1) for i in range(6)])
