import typing

import torch
from poke_env.environment import Battle
from poke_env.environment import DoubleBattle
from poke_env.environment import Move
from poke_env.environment import Pokemon
from poke_env.player import BattleOrder
from poke_env.player import ForfeitBattleOrder
from poke_env.player import Player
from stable_baselines3.common.base_class import BaseAlgorithm

from champion_league.preprocessor import Preprocessor


class OpponentPlayer(Player):
    def __init__(
        self,
        preprocessor: Preprocessor,
        model: BaseAlgorithm,
        name: str,
        *args,
        **kwargs
    ):
        super(OpponentPlayer, self).__init__(*args, **kwargs)
        self._preprocessor = preprocessor
        self._model = model
        self.name = name

    def choose_move(self, battle: Battle) -> BattleOrder:
        action, _ = self._model.predict(
            torch.from_numpy(self._preprocessor.embed_battle(battle)).view(1, -1)
        )
        return self.action_to_move(int(action), battle)

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:  # pyre-ignore
        """Converts actions to move orders.

        The conversion is done as follows:

        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 12], dynamax=True)
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return self.choose_random_move(battle)

    @staticmethod
    def create_order(
        order: typing.Union[Move, Pokemon],
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
