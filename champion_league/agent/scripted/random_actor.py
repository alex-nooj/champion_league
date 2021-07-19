import random

from adept.utils.util import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from champion_league.agent.scripted.base_scripted import BaseScripted


class RandomActor(BaseScripted):
    def choose_move(self, battle) -> BattleOrder:
        return self.choose_random_move(battle)
