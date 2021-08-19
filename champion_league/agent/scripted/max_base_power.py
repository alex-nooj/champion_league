from champion_league.agent.scripted.base_scripted import BaseScripted


class MaxBasePower(BaseScripted):
    def choose_move(self, battle):
        """Chooses a move for the agent. Determines the move that has the highest power.

        Parameters
        ----------
        battle: Battle
            The raw battle output from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
