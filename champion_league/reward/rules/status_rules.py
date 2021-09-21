from poke_env.environment.battle import Battle

from champion_league.reward.rules.rule import Rule


class OpponentStatusRule(Rule):
    def __init__(self, weight: float):
        super().__init__(weight=weight)
        self._status_count = 0

    def compute(self, curr_step: Battle) -> float:
        status_count = len(
            [None for mon in curr_step.opponent_team.values() if mon.status is not None]
        )
        reward = self.weight * int(status_count > self._status_count)
        self._status_count = status_count
        return reward

    def reset(self):
        self._status_count = 0

    @property
    def max(self) -> float:
        return self.weight


class AlliedStatusRule(OpponentStatusRule):
    def compute(self, curr_step: Battle) -> float:
        status_count = len(
            [None for mon in curr_step.team.values() if mon.status is not None]
        )
        reward = -1 * self.weight * int(status_count > self._status_count)
        self._status_count = status_count
        return reward
