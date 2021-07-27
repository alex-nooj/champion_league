import numpy as np

from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder

from champion_league.agent.scripted.base_scripted import BaseScripted


class AlwaysSwitches(BaseScripted):
    def choose_move(self, battle: Battle) -> BattleOrder:
        if len(battle.available_switches) != 0:
            return BattleOrder(np.random.choice(battle.available_switches))
        else:
            return self.choose_random_move(battle)
