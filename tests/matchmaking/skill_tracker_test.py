import os
from typing import Dict

import pytest

from champion_league.training.league.league_skill_tracker import LeagueSkillTracker


@pytest.mark.parametrize(
    "agent_won,opponent,new_agent_skill,new_opponent_skill",
    [
        (
            True,
            "agent0",
            {"mu": 29.396, "sigma": 7.171},
            {"mu": 20.604, "sigma": 7.171},
        ),
        (
            False,
            "agent0",
            {"mu": 20.604, "sigma": 7.171},
            {"mu": 29.396, "sigma": 7.171},
        ),
    ],
)
def test_update(
    agent_won: bool,
    opponent: str,
    new_agent_skill: Dict[str, float],
    new_opponent_skill: Dict[str, float],
):
    skill_tracker = LeagueSkillTracker(
        tag="test_skill_tracker",
        logdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data"),
        default_mu=25,
        default_sigma=8.333,
        resume=False,
    )

    skill_tracker.update(agent_won=agent_won, opponent=opponent)

    agent_skill = skill_tracker.skill
    opponent_skill = {
        "mu": skill_tracker.skill_ratings[opponent].mu,
        "sigma": skill_tracker.skill_ratings[opponent].sigma,
    }

    assert round(agent_skill["mu"], 3) == new_agent_skill["mu"]
    assert round(agent_skill["sigma"], 3), new_agent_skill["sigma"]
    assert round(opponent_skill["mu"], 3), new_opponent_skill["mu"]
    assert round(opponent_skill["sigma"], 3), new_opponent_skill["sigma"]
