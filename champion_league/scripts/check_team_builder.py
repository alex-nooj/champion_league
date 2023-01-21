import asyncio
import pathlib

import trueskill

from champion_league.env import LeaguePlayer
from champion_league.training.common import MatchMaker
from champion_league.training.league.league_team_builder import LeagueTeamBuilder


class SimpleMatchMaker(MatchMaker):
    def __init__(self, scripted: bool):
        self.scripted = scripted

    def choose_match(self, *args, **kwargs):
        if self.scripted:
            self.scripted = False
            return pathlib.Path(
                "/home/anewgent/Projects/pokemon/sinnoh_league/league/Red_00000"
            )
        else:
            self.scripted = True
            return pathlib.Path(
                "/home/anewgent/Projects/pokemon/sinnoh_league/league/Blue_00039"
            )


if __name__ == "__main__":
    player1 = LeaguePlayer(
        device=0,
        matchmaker=SimpleMatchMaker(True),
        team=LeagueTeamBuilder(),
        battle_format="gen8ou",
        max_concurrent_battles=1,
    )

    player2 = LeaguePlayer(
        device=0,
        matchmaker=SimpleMatchMaker(False),
        team=LeagueTeamBuilder(),
        battle_format="gen8ou",
        max_concurrent_battles=1,
    )

    nb_battles = 10
    for _ in range(nb_battles):
        player1.change_agent(trueskill.Rating(), {})
        player2.change_agent(trueskill.Rating(), {})
    asyncio.get_event_loop().run_until_complete(
        player1.battle_against(player2, n_battles=10)
    )
