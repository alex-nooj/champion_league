import random
from typing import Union

from champion_league.utils.directory_utils import DotDict
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder


class BaseScripted:
    def __init__(self, tag: str):
        self.tag = tag

    def choose_random_move(self, battle: AbstractBattle) -> BattleOrder:
        """Chooses a random move from the four moves the pokemon knows, or one of the available
        pokemon to switch to.

        Parameters
        ----------
        battle: AbstractBattle
            The raw battle from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        if isinstance(battle, Battle):
            return self.choose_random_singles_move(battle)
        else:
            raise ValueError(
                "battle should be Battle or DoubleBattle. Received %d" % (type(battle))
            )

    @staticmethod
    def choose_random_singles_move(battle: Battle) -> BattleOrder:
        """Helps choose a random move, ignoring Z moves, mega-evolution, and dynamaxing.

        Parameters
        ----------
        battle: Battle
            The raw battle from the environment.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
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

        Parameters
        ----------
        order: Union[Move, Pokemon]
            Move to make or Pokemon to switch to.
        mega: bool
            Whether to mega evolve the pokemon, if a move is chosen.
        z_move: bool
            Whether to make a zmove, if a move is chosen.
        dynamax: bool
            Whether to dynamax, if a move is chosen.
        move_target: int
            Target Pokemon slot of a given move.

        Returns
        -------
        BattleOrder
            The order the agent wants to take, converted into a form readable by the environment.
        """
        return BattleOrder(
            order, mega=mega, move_target=move_target, z_move=z_move, dynamax=dynamax
        )
