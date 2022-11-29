import os

import pytest
import trueskill

from champion_league.training.league.league_matchmaker import LeagueMatchMaker


class TestMatchMaker:
    @pytest.mark.parametrize("test_len", (1000,))
    def test_choose_self(self, test_len: int):
        matchmaker = LeagueMatchMaker(
            self_play_prob=1.0,
            league_play_prob=0.0,
            logdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            tag="challenger_0",
        )

        t1 = trueskill.Rating()
        t2 = trueskill.Rating()
        nb_of_selves = 0
        for _ in range(test_len):
            nb_of_selves += int(matchmaker.choose_match(t1, {"t2": t2}) == "self")

        assert nb_of_selves / test_len >= 0.95
