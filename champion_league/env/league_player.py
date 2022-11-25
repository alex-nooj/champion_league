import copy
import pathlib
import random
import typing

import torch
from omegaconf import OmegaConf
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.player.player import Player
from torch import nn

from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.preprocessor import Preprocessor
from champion_league.teams.team_builder import AgentTeamBuilder


class LeaguePlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        device: int,
        network: nn.Module,
        preprocessor: Preprocessor,
        sample_moves: typing.Optional[bool] = True,
        team: typing.Optional[AgentTeamBuilder] = None,
        training_team: typing.Optional[AgentTeamBuilder] = None,
        **kwargs: typing.Any,
    ):
        """This is the player for the league. It acts as the opponent for the training agent and
        handles all server communications.

        Args:
            device: The device to load the networks onto.
            network: The network that is currently training, used for self-play.
            preprocessor: The preprocessor that the agent is using.
            sample_moves: Whether to sample the output distribution of the network.
            kwargs: Additional keyword arguments. Any of those used by PokeEnv would be placed here
        """
        super().__init__(team=team, **kwargs)
        self.sample_moves = sample_moves

        self.scripted_agent = None
        self.mode = None
        self.device = device
        self.training_network = None
        self.training_preprocessor = preprocessor
        self.network = None
        self.preprocessor = None
        self.update_network(network)
        self.training_team = training_team
        self.team = team
        self.tag = self.change_agent("self")

    def choose_move(self, battle: Battle) -> BattleOrder:
        """The function used to pass the current state into the network or scripted agent and
        receive a battle order.

        Args:
            battle: The current state in its raw form.

        Raises:
            RuntimeError: If an agent has not been selected yet and the server asks for a move.

        Returns:
            BattleOrder: The move that the agent would like to select
        """
        if self.mode is None:
            raise RuntimeError("Agent cannot be none!")

        if self.mode != "scripted":
            state = self.preprocessor.embed_battle(battle)

            with torch.no_grad():
                y = self.network(x=state)

            if self.sample_moves:
                action = torch.multinomial(y["action"][0:], 1).item()
            else:
                action = torch.argmax(y["action"][0:], dim=-1).item()

            if (
                action < 4
                and action < len(battle.available_moves)
                and not battle.force_switch
            ):
                return BattleOrder(battle.available_moves[action])
            elif 0 <= action - 4 < len(battle.available_switches):
                return BattleOrder(battle.available_switches[action - 4])
            else:
                return self.choose_random_move(battle)

        return self.scripted_agent.choose_move(battle)

    def change_agent(
        self,
        agent_path: typing.Union[str, pathlib.Path],
    ) -> str:
        """This handles the swapping of the agent choosing the moves. For the league, this is useful
        as it allows the agent to see many selection strategies during training, making it
        more robust. From a software engineering standpoint, this also allows us to switch agents
        really quickly while maintaining the current connection to the server.

        NOTE: LeaguePlayer does not handle the logic of which agent to select, it just switches to
        the desired agent.

        Args:
            agent_path: The path to the chosen agent.

        Returns:
            str: The name of the current agent, or 'self' if we are currently running self-play
        """

        if agent_path == "self":
            # If we're doing self-play, then this loads up an opponent that is a copy of the current
            # network.
            self.network = self.training_network
            self.preprocessor = self.training_preprocessor
            self.mode = "self"
            self.tag = "self"
            self._team = self.training_team
        else:
            # Otherwise, we're playing a league agent, so we have to build that network. So first we
            # load up the arguments, which will act as build instructions.

            args = OmegaConf.to_container(
                OmegaConf.load(pathlib.Path(agent_path, "args.yaml"))
            )
            if "scripted" in args:
                # Scripted agents act differently than ML agents, so we have to treat them a little
                # differently.
                self.mode = "scripted"
                self.scripted_agent = SCRIPTED_AGENTS[args["tag"]]
                self.network = None
                self.preprocessor = None
                self._team = self.team
                self._team.clear_team()
            else:
                # Otherwise, we have an ML agent, and have to build the LeagueOpponent class using
                # this network as a selection strategy.
                self.mode = "ml"
                agent_data = torch.load(
                    pathlib.Path(agent_path, "network.pth"), map_location=self.device
                )
                self.network = agent_data["network"]
                self.preprocessor = agent_data["preprocessor"]
                self._team = agent_data["team"]

            # Sometimes, we aren't actually fighting a league opponent and are instead just fighting
            # an earlier iteration of the current agent, which should still count as self-play
            if list(agent_path.parents)[2] == "challengers":
                self.tag = "self"
            else:
                self.tag = agent_path.stem

        return self.tag

    def update_network(self, network: nn.Module) -> None:
        """Performs a deep copy of the network. Useful for handling self-play.

        Args:
            network: The network that should replace the current one stored by the agent.
        """
        self.training_network = copy.deepcopy(network).eval()

    def reset(self):
        self.preprocessor.reset()

    def teampreview(self, battle):
        return "/team " + "".join([str(i + 1) for i in range(6)])

    def choose_random_move(self, battle: Battle) -> BattleOrder:
        """This allows the agent to choose a random move when the order it would like is unavailable

        Args:
            battle: The current, raw state of the Pokemon battle.

        Returns:
            BattleOrder: The selected action that is readable by the environment.
        """
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        else:
            return DefaultBattleOrder()
