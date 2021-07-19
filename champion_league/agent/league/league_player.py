import importlib
import json
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import numpy as np
import torch.nn.functional as F
import torch
from adept.utils.util import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from champion_league.agent.scripted.simple_heuristic import SimpleHeuristic
from champion_league.agent.scripted.max_base_power import MaxBasePower
from champion_league.agent.scripted.random_actor import RandomActor
from champion_league.agent.league.eval_agent import EvalAgent


class LeaguePlayer(Player):
    def __init__(
        self,
        args: DotDict,
        player_configuration: Optional[PlayerConfiguration] = None,
        *,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 2,
        server_configuration: Optional[ServerConfiguration] = None,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        team: Optional[Union[str, Teambuilder]] = None,
    ) -> None:
        """Player class that will act as the league. Whenever the game ends, call change_agent() to
        change the agent playing the game. Allows sampling of self-play, league-play, and exploiting

        Parameters
        ----------
        args: DotDict
            DotDict containing:
            - opponent_device: int
                Device for this agent to use
            - p_exploit: float
                How often the agent should be an exploiter (between 0.0 and 1.0)
            - p_league: float
                How often the agent should be from the league
            - logdir
                The path to where main agents, league agents, and exploiters are kept
            - tag
                The name of the main agent that is currently training
        player_configuration: Optional[PlayerConfiguration]
            Player configuration. If empty, defaults to an
            automatically generated username with no password. This option must be set
            if the server configuration requires authentication.
        avatar: Optional[int]
            Player avatar id. Optional.
        battle_format: Optional[str]
            Name of the battle format this player plays. Defaults to
            gen8randombattle.
        log_level: Optional[int]
            The player's logger level.
        max_concurrent_battles: Optional[int]
            Maximum number of battles this player will play concurrently. If 0, no limit will be
            applied.
        server_configuration: Optional[ServerConfiguration]
            Server configuration. Defaults to Localhost Server Configuration
        start_timer_on_battle_start: bool
            Whether or not to start the battle timer
        start_listening: bool
            Whether to start listening to the server. Defaults to True
        team: Optional[Union[str, Teambuilder]]
            The team to use for formats requiring a team. Can be a showdown team string, a showdown
            packed team string, or a ShowdownTeam object. Defaults to None.
        """
        self.args = args
        self._device = args.opponent_device
        self._scripted = False
        self.p_exploit = 0
        self.p_league = 0
        self.internals = {"hx": None, "cx": None}
        self.current_agent = None

        self.change_agent({})
        self.p_exploit = int(args.p_exploit * 100)
        self.p_league = int((args.p_exploit + args.p_league) * 100)

        super().__init__(
            player_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            team=team,
        )

    @staticmethod
    def sample_agents(win_rates: Dict[str, List[int]], possible_agents: List[str]) -> str:
        """Determines what agent to play as, given a list of agents played, their win rates, and the
        list of possible agents. This function will prioritize agents that have not yet been played
        (do not appear in win_rates)

        Parameters
        ----------
        win_rates: Dict[str, List[int]]
            Dict with keys referring to agent tags, i.e. AGENTNAME_EPOCH and a list with entry 0
            being number of wins agains the key agent, and entry 1 being the number of games played
            against the key agent
        possible_agents: List[str]
            List of agent names to choose from

        Returns
        -------
        str
            Name of the agent to play against
        """
        if all(name in win_rates for name in possible_agents):
            percent_wins = torch.tensor(
                [win_rates[name][0] / win_rates[name][1] for name in possible_agents]
            )
            distribution = 1.0 - percent_wins
            agent_ix = torch.multinomial(F.softmax(distribution, dim=0), 1).item()
            return possible_agents[agent_ix]
        else:
            unseen_opponents = [name for name in possible_agents if name not in win_rates]
            return unseen_opponents[0]

    def change_agent(self, win_rates: Dict[str, List[int]]):
        """Changes the agent currently playing the game

        Parameters
        ----------
        win_rates: Dict[str, List[int]]
            Dict with keys referring to agent tags, i.e. AGENTNAME_EPOCH and a list with entry 0
            being number of wins agains the key agent, and entry 1 being the number of games played
            against the key agent

        Returns
        -------
        None
        """
        curr_mode = np.random.randint(low=0, high=100)

        if os.path.isdir(os.path.join(self.args.logdir, "exploiters", self.args.tag)):
            exploiters = os.listdir(os.path.join(self.args.logdir, "exploiters", self.args.tag))
        else:
            exploiters = []
        league_agents = os.listdir(os.path.join(self.args.logdir, "league"))
        previous_selves = [
            agent_path
            for agent_path in os.listdir(
                os.path.join(self.args.logdir, "challengers", self.args.tag)
            )
            if self.args.tag in agent_path
        ]

        if curr_mode < self.p_exploit and len(exploiters) != 0:
            agent = self.sample_agents(win_rates, exploiters)
            agent_path = os.path.join(self.args.logdir, "exploiters", self.args.tag, agent)
        elif curr_mode < self.p_league and len(league_agents):
            agent = self.sample_agents(win_rates, league_agents)
            agent_path = os.path.join(self.args.logdir, "league", agent)
        else:
            agent = self.sample_agents(win_rates, previous_selves)
            agent_path = os.path.join(self.args.logdir, "challengers", self.args.tag, agent)

        with open(os.path.join(agent_path, "args.json"), "r") as fp:
            agent_args = json.load(fp)
            agent_args = DotDict(agent_args)
        agent_args.device = self._device

        if agent_args.tag == "max_base_power_0":
            self.current_agent = MaxBasePower(agent_args)
        elif agent_args.tag == "random_0":
            self.current_agent = RandomActor(agent_args)
        elif agent_args.tag == "simple_heuristic_0":
            self.current_agent = SimpleHeuristic(agent_args)
        else:
            self.internals = {"hx": torch.zeros((1, 512)), "cx": torch.zeros((1, 512))}
            self.current_agent = EvalAgent(agent_args, agent, agent_path)

    def choose_move(self, battle: Battle) -> BattleOrder:
        """Implementation of abstract method. Uses the current agent to select a move

        Parameters
        ----------
        battle: AbstractBattle
            The current state
        Returns
        -------
        BattleOrder
            The chosen move
        """
        if any(
            tag == self.current_agent.tag
            for tag in ["max_base_power_0", "random_0", "simple_heuristic_0"]
        ):
            return self.current_agent.choose_move(battle)
        else:
            move, self.internals = self.current_agent.choose_move(battle, self.internals)
            return self._action_to_move(move, battle)

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:  # pyre-ignore
        """Converts actions to move orders.

        The conversion is done as follows:

        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action < 4 and action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 12], dynamax=True)
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return self.choose_random_move(battle)
