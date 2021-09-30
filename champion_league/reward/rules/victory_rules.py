from poke_env.environment.battle import Battle

from champion_league.reward.rules.rule import Rule


class VictoryRule(Rule):
    """Rule for determine reward based on if the agent has won."""

    def compute(self, curr_step: Battle) -> float:
        """Computes the reward for the given step of the battle based on if the agent won.

        Parameters
        ----------
        curr_step: Battle
            The current state of the battle.

        Returns
        -------
        float
            self.weight if the agent has won, 0 otherwise.
        """
        return self.weight * (curr_step.won is not None)

    @property
    def max(self) -> float:
        """The max value for this rule.

        Returns
        -------
        float
            The maximum value for this rule.
        """
        return self.weight

    def reset(self):
        """The reset function for the rule."""
        pass


class LossRule(Rule):
    """Rule for determine reward based on if the agent has lost."""

    def compute(self, curr_step: Battle) -> float:
        """Computes the reward for the given step of the battle based on if the agent lost.

        Parameters
        ----------
        curr_step: Battle
            The current state of the battle.

        Returns
        -------
        float
            -1 * self.weight if the agent has won, 0 otherwise.
        """
        return -1 * self.weight * curr_step.lost

    @property
    def max(self) -> float:
        """The max value for this rule.

        Returns
        -------
        float
            The maximum value for this rule.
        """
        return self.weight

    def reset(self):
        """The reset function for the rule."""
        pass
