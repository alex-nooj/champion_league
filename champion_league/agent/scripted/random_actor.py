from poke_env.player.battle_order import BattleOrder

from champion_league.agent.scripted.base_scripted import BaseScripted


class RandomActor(BaseScripted):
    def choose_move(self, battle) -> BattleOrder:
        """Randomly chooses a move for the agent.

        Parameters
        ----------
        battle: Battle
            The raw battle output from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        return self.choose_random_move(battle)
