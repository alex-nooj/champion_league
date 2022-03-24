import os
from typing import Optional

from poke_env.teambuilder.teambuilder import Teambuilder

from champion_league.utils.random_team_generator import generate_random_team


def load_team_from_file(path: str) -> str:
    with open(os.path.join(path, "team.txt"), "r") as fp:
        team = fp.read()
    return team


def save_team_to_file(path: str, team: str):
    with open(os.path.join(path, "team.txt"), "w") as fp:
        fp.write(team)


class AgentTeamBuilder(Teambuilder):
    def __init__(
        self, path: Optional[str] = None, battle_format: Optional[str] = "gen8ou"
    ):
        if path is not None:
            try:
                self._team = self.join_team(
                    self.parse_showdown_team(load_team_from_file(path))
                )
            except FileNotFoundError:
                team = generate_random_team(battle_format=battle_format)
                self._team = self.join_team(self.parse_showdown_team(team))
                save_team_to_file(path, team)

        else:
            self._team = None

    def yield_team(self) -> str:
        if self._team is not None:
            return self._team
        else:
            return self.join_team(self.parse_showdown_team(generate_random_team()))

    def load_new_team(self, path: str):
        self._team = self.join_team(self.parse_showdown_team(load_team_from_file(path)))

    def clear_team(self):
        self._team = None

    def set_team(self, team: str):
        self._team = self.join_team(self.parse_showdown_team(team))
