from poke_env.environment.battle import Battle

from champion_league.reward.rules.binary_faints import count_faints
from champion_league.reward.rules.rule import Rule

NB_POKEMON = 12


class OpponentScalarFaints(Rule):
    def __init__(self, weight: float):
        """Rule for calculating reward based on if the agent has fainted an opponent's pokemon.

        Parameters
        ----------
        weight: float
            How much to weight this reward scheme by.
        """
        super().__init__(weight=weight)
        self.prev_fainted_pokemon = 0

    def compute(self, curr_step: Battle) -> float:
        """Computes the reward for the given step of the battle based on if an opponent's pokemon
        has fainted this turn.

        Parameters
        ----------
        curr_step: Battle
            The current state of the battle.

        Returns
        -------
        float
            The difference between the fainted pokemon in the previous step and this step.
        """
        curr_fainted_pokemon = count_faints(curr_step.opponent_team)

        reward = self.weight * (curr_fainted_pokemon - self.prev_fainted_pokemon)
        self.prev_fainted_pokemon = curr_fainted_pokemon

        return reward

    @property
    def max(self) -> float:
        """The max value for this rule.

        Returns
        -------
        float
            The maximum value for this rule.
        """
        return self.weight * (NB_POKEMON // 2)

    def reset(self):
        """The reset function for the rule."""
        self.prev_fainted_pokemon = 0


class AlliedScalarFaints(OpponentScalarFaints):
    """Rule for calculating reward based on if the agent has fainted an opponent's pokemon."""

    def compute(self, curr_step: Battle) -> float:
        """Computes the reward for the given step of the battle based on if an allied pokemon
        has fainted this turn.

        Parameters
        ----------
        curr_step: Battle
            The current state of the battle.

        Returns
        -------
        float
            The difference between the fainted pokemon in this step and the previous.
        """
        curr_fainted_pokemon = count_faints(curr_step.opponent_team)

        reward = self.weight * (self.prev_fainted_pokemon - curr_fainted_pokemon)
        self.prev_fainted_pokemon = curr_fainted_pokemon

        return reward
