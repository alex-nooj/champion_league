import collections
import typing

from poke_env.teambuilder.teambuilder import Teambuilder

from champion_league.utils.random_team_generator import generate_random_team


class LeagueTeamBuilder(Teambuilder):
    def __init__(self, battle_format: typing.Optional[str] = "genou"):
        self.battle_format = battle_format
        self._teams = collections.deque()

    def yield_team(self) -> str:
        return self._teams.popleft()

    def add_to_stack(self, team: str):
        self._teams.append(team)

    def add_random_to_stack(self):
        self._teams.append(
            self.join_team(
                self.parse_showdown_team(
                    generate_random_team(battle_format=self.battle_format)
                )
            )
        )
