from typing import Dict

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from champion_league.reward.rules.rule import Rule


def hp_total(team: Dict[str, Pokemon]) -> int:
    return sum([p.current_hp_fraction for p in team.values()])


class OpponentHPDiff(Rule):
    def __init__(self, weight: float):
        super().__init__(weight=weight)
        self.prev_hp_total = 0

    def compute(self, curr_step: Battle) -> float:
        curr_hp_total = hp_total(curr_step.opponent_team)

        reward = self.weight * (self.prev_hp_total - curr_hp_total)
        self.prev_hp_total = curr_hp_total

        return reward

    @property
    def max(self) -> float:
        return self.weight

    def reset(self):
        self.prev_hp_total = 0


class AlliedHPDiff(OpponentHPDiff):
    def compute(self, curr_step: Battle) -> float:
        curr_hp_total = hp_total(curr_step.team)
        reward = self.weight * (self.prev_hp_total - curr_hp_total)
        self.prev_hp_total = curr_hp_total

        return reward
