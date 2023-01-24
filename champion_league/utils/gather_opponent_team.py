import typing

from poke_env.environment import AbstractBattle
from poke_env.environment.pokemon import Pokemon


def gather_opponent_team(battle: AbstractBattle) -> typing.List[Pokemon]:
    opponent_team = [battle.opponent_active_pokemon]
    for mon in battle._teampreview_opponent_team:
        if mon.species not in [m.species for m in opponent_team]:
            opponent_team.append(mon)
    return opponent_team