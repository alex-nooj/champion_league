from poke_env.player.battle_order import BattleOrder

from champion_league.agent.scripted.base_scripted import BaseScripted


class RandomActor(BaseScripted):
    def choose_move(self, battle) -> BattleOrder:
        return self.choose_random_move(battle)
