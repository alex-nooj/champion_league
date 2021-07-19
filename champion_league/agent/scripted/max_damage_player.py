from adept.utils.util import DotDict
from poke_env.player.player import Player


class MaxDamagePlayer(Player):
    BATTLES = {}

    def __init__(self, args: DotDict):
        super().__init__()

    def choose_move(self, battle):
        if battle.battle_tag not in self.BATTLES:
            self.BATTLES[battle.battle_tag] = [battle]
        else:
            self.BATTLES[battle.battle_tag].append(battle)

        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)
