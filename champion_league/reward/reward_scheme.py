from typing import Dict

from poke_env.environment.battle import Battle

from champion_league.reward.rules import RULES
from champion_league.reward.rules.rule import Rule


class RewardScheme(Rule):
    def __init__(self, rules: Dict[str, float]):
        """Class for handling multiple rules for rewards. Also acts as a rule itself.

        Parameters
        ----------
        rules: Dict[str, float]
            A dictionary containing the names of the desired rules as keys and the desired weights
            for each of those rules as values.
        """
        super().__init__(weight=1.0)
        self.rules = [RULES[rule](weight) for rule, weight in rules.items()]
        self.prev_step = None
        self._max = sum([r.max for r in self.rules])

    def reset(self):
        """Reset function for the reward scheme."""
        self.prev_step = None

        for rule in self.rules:
            rule.reset()

    def compute(self, step: Battle) -> float:
        """Computes the reward for the given step of the battle.

        Parameters
        ----------
        step: Battle
            The current state of the battle.

        Returns
        -------
        float
            The reward for `step`
        """
        reward = 0
        for rule in self.rules:
            reward += rule.compute(
                curr_step=step,
            )
        self.prev_step = step
        return reward

    @property
    def max(self) -> float:
        """The max value of the reward scheme.

        Returns
        -------
        float
            The maximum value of the reward scheme.
        """
        return self._max
