from typing import Dict

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from champion_league.reward.rules.rule import Rule


def count_faints(team: Dict[str, Pokemon]) -> int:
    return len([None for p in team.values() if p.fainted])


class OpponentBinaryFaints(Rule):
    def __init__(self, weight: float):
        super().__init__(weight=weight)
        self.prev_fainted_pokemon = 0

    def compute(self, curr_step: Battle) -> float:
        curr_fainted_pokemon = count_faints(curr_step.opponent_team)

        reward = self.weight * int(curr_fainted_pokemon != self.prev_fainted_pokemon)
        self.prev_fainted_pokemon = curr_fainted_pokemon

        return reward

    @property
    def max(self) -> float:
        return self.weight

    def reset(self):
        self.prev_fainted_pokemon = 0


class AlliedBinaryFaints(OpponentBinaryFaints):
    def compute(self, curr_step: Battle) -> float:
        curr_fainted_pokemon = count_faints(curr_step.team)

        reward = -self.weight * int(curr_fainted_pokemon != self.prev_fainted_pokemon)
        self.prev_fainted_pokemon = curr_fainted_pokemon

        return reward
