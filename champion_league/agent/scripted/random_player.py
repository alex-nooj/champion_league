from poke_env.player.player import Player

from champion_league.utils.directory_utils import DotDict


class RandomPlayer(Player):
    BATTLES = {}

    def __init__(self, args: DotDict):
        super().__init__()

    def choose_move(self, battle):
        """Chooses a move for the agent.

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
