import collections
import pathlib
import random
import typing

import torch
import trueskill
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.player.player import Player
from torch import nn

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.scripted import SCRIPTED_AGENTS
from champion_league.preprocessor import Preprocessor
from champion_league.training.common import MatchMaker
from champion_league.training.league.league_team_builder import LeagueTeamBuilder


def scripted(tag: typing.Union[str, pathlib.Path]) -> bool:
    if isinstance(tag, str):
        parent_dir = pathlib.Path(tag).stem
    else:
        parent_dir = tag.stem
    return any(
        name == parent_dir
        for name in ["random_0", "max_base_power_0", "simple_heuristic_0"]
    )


class LeaguePlayer(Player):
    BATTLES = {}

    def __init__(
        self,
        device: int,
        matchmaker: MatchMaker,
        team: typing.Optional[LeagueTeamBuilder] = None,
        **kwargs: typing.Any,
    ):
        """This is the player for the league. It acts as the opponent for the training agent and
        handles all server communications.

        Args:
            device: The device to load the networks onto.
            kwargs: Additional keyword arguments. Any of those used by PokeEnv would be placed here
        """
        if team is None:
            team = LeagueTeamBuilder()

        super().__init__(team=team, **kwargs)
        self.device = torch.device(f"cuda:{device}")
        self.matchmaker = matchmaker
        self.network = None
        self.preprocessor = None
        self._next_opponent = collections.deque()
        self.tag = None
        self.scripted_agent = None
        self.mode = None
        self.team = None

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
            timeout = 0
            while not isinstance(self.preprocessor, Preprocessor):
                timeout += 1
            state = self.preprocessor.embed_battle(battle)

            with torch.no_grad():
                y = self.network(x=state)
                action = torch.argmax(y["rough_action"][0:], dim=-1).item()

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
        agent_skill: trueskill.Rating,
        trueskills: typing.Dict[str, trueskill.Rating],
    ) -> str:
        """This handles the swapping of the agent choosing the moves. For the league, this is useful
        as it allows the agent to see many selection strategies during training, making it
        more robust. From a software engineering standpoint, this also allows us to switch agents
        really quickly while maintaining the current connection to the server.

        Args:
            agent_skill: The trueskill rating of the training agent.
            trueskills: The trueskill ratings of the league agents.

        Returns:
            str: The name of the current agent, or 'self' if we are currently running self-play
        """

        actor, preprocessor, self.tag, team = self._choose_agent(
            agent_skill, trueskills
        )

        if preprocessor is None:
            # Scripted agents act differently than ML agents, so we have to treat them a little
            # differently.
            self.mode = "scripted"
            self.scripted_agent = actor
        else:
            # Otherwise, we have an ML agent, and have to build the LeagueOpponent class using
            # this network as a selection strategy.
            self.mode = "ml"
            self.network = actor.to(self.device)
            self.preprocessor = preprocessor
            self.team = team
        return self.tag

    def reset(self):
        if self.preprocessor is not None:
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

    def _choose_agent(
        self,
        agent_skill: trueskill.Rating,
        trueskills: typing.Dict[str, trueskill.Rating],
    ) -> typing.Tuple[
        typing.Union[Agent, nn.Module],
        typing.Union[Preprocessor, None],
        str,
        typing.Union[str, None],
    ]:
        while len(self._next_opponent) < 4:
            opponent = self.matchmaker.choose_match(agent_skill, trueskills)
            if scripted(opponent):
                self._next_opponent.append(
                    (SCRIPTED_AGENTS[opponent.stem], None, opponent.stem, None)
                )
                self._team.add_random_to_stack()
            else:
                agent_data = torch.load(opponent / "network.pth", map_location="cpu")
                self._next_opponent.append(
                    (
                        agent_data["network"],
                        agent_data["preprocessor"],
                        opponent.stem,
                        agent_data["team"].yield_team(),
                    )
                )
                self._team.add_to_stack(agent_data["team"].yield_team())

        return self._next_opponent.popleft()
