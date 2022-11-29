import pathlib

from champion_league.training.common import MatchMaker


class AgentMatchMaker(MatchMaker):
    def __init__(self, opponent_path: pathlib.Path):
        self.opponent_path = opponent_path

    def choose_match(self, *args, **kwargs) -> pathlib.Path:
        return self.opponent_path
