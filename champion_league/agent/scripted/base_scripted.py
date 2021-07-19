import random

from adept.utils.util import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from typing import Union


class BaseScripted:
    def __init__(self, tag: str):
        self.tag = tag

    def choose_random_move(self, battle: AbstractBattle) -> BattleOrder:
        """Returns a random legal move from battle.

        :param battle: The battle in which to move.
        :type battle: AbstractBattle
        :return: Move order
        :rtype: str
        """
        if isinstance(battle, Battle):
            return self.choose_random_singles_move(battle)
        else:
            raise ValueError(
                "battle should be Battle or DoubleBattle. Received %d" % (type(battle))
            )

    @staticmethod
    def choose_random_singles_move(battle: Battle) -> BattleOrder:
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend([BattleOrder(switch) for switch in battle.available_switches])

        if battle.can_mega_evolve:
            available_orders.extend(
                [BattleOrder(move, mega=True) for move in battle.available_moves]
            )

        if battle.can_dynamax:
            available_orders.extend(
                [BattleOrder(move, dynamax=True) for move in battle.available_moves]
            )

        if battle.can_z_move and battle.active_pokemon:
            available_z_moves = set(battle.active_pokemon.available_z_moves)  # pyre-ignore
            available_orders.extend(
                [
                    BattleOrder(move, z_move=True)
                    for move in battle.available_moves
                    if move in available_z_moves
                ]
            )

        if available_orders:
            return available_orders[int(random.random() * len(available_orders))]
        else:
            return DefaultBattleOrder()

    @staticmethod
    def create_order(
        order: Union[Move, Pokemon],
        mega: bool = False,
        z_move: bool = False,
        dynamax: bool = False,
        move_target: int = DoubleBattle.EMPTY_TARGET_POSITION,
    ) -> BattleOrder:
        """Formats an move order corresponding to the provided pokemon or move.

        :param order: Move to make or Pokemon to switch to.
        :type order: Move or Pokemon
        :param mega: Whether to mega evolve the pokemon, if a move is chosen.
        :type mega: bool
        :param z_move: Whether to make a zmove, if a move is chosen.
        :type z_move: bool
        :param dynamax: Whether to dynamax, if a move is chosen.
        :type dynamax: bool
        :param move_target: Target Pokemon slot of a given move
        :type move_target: int
        :return: Formatted move order
        :rtype: str
        """
        return BattleOrder(
            order, mega=mega, move_target=move_target, z_move=z_move, dynamax=dynamax
        )
