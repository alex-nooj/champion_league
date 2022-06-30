from pathlib import Path
from typing import Optional

from poke_env.teambuilder.teambuilder import Teambuilder

from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.random_team_generator import generate_random_team


def load_team_from_file(file_path: Path) -> str:

    with open(str(file_path / "team.txt"), "r") as fp:
        team = fp.read()
    return team


def save_team_to_file(file_path: Path, team: str):
    with open(str(file_path / "team.txt"), "w") as fp:
        fp.write(team)


class AgentTeamBuilder(Teambuilder):
    def __init__(
        self, agent_path: Optional[Path] = None, battle_format: Optional[str] = "gen8ou"
    ):
        self.agent_path = agent_path
        if agent_path is not None:
            try:
                self._team = self.join_team(
                    self.parse_showdown_team(load_team_from_file(agent_path))
                )
            except FileNotFoundError:
                team = generate_random_team(battle_format=battle_format)
                self._team = self.join_team(self.parse_showdown_team(team))
                save_team_to_file(agent_path, team)

        else:
            self._team = None

    def yield_team(self) -> str:
        if self._team is not None:
            return self._team
        else:
            return self.join_team(self.parse_showdown_team(generate_random_team()))

    def load_new_team(self, agent_path: Path):
        self._team = self.join_team(
            self.parse_showdown_team(load_team_from_file(agent_path))
        )

    def clear_team(self):
        self._team = None

    def set_team(self, team: str):
        self._team = self.join_team(self.parse_showdown_team(team))

    def save_team(self):
        if self.agent_path is None:
            raise RuntimeError("Agent path cannot be none when saving team!")
        elif self._team is None:
            raise RuntimeError("Team cannot be none when saving team!")
        else:
            save_team_to_file(
                file_path=get_save_dir(
                    agent_dir=self.agent_path,
                    epoch=get_most_recent_epoch(self.agent_path),
                ),
                team=self._team,
            )
