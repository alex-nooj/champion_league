import asyncio

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.status import Status
from poke_env.utils import to_id_str
from typing import Optional, Union, Tuple

from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from champion_league.env.base.obs_idx import ObsIdx
from champion_league.utils.abilities import ABILITIES


class PokemonBattleEnv:
    _ACTINO_SPACE = list(range(4 * 4 + 6))
    _DEFAULT_BATTLE_FORMAT = "gen8randombattle"
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001
    MESSAGES_TO_IGNORE = {"", "t:", "expire"}

    DEFAULT_CHOICE_CHANCE = 1 / 1000

    def __init__(
            self,
            player_configuration: Optional[PlayerConfiguration] = None,
            *,
            battle_format: Optional[str] = None,
            log_level: Optional[int] = None,
            server_configuration: Optional[ServerConfiguration] = None,
            start_listening: bool = True,
            start_timer_on_battle_start: bool = False,
            team: Optional[Union[str, Teambuilder]] = None,
    ):
        """

        Parameters
        ----------
        player_configuration
        battle_format
        log_level
        server_configuration
        start_listening
        start_timer_on_battle_start
        team
        """
    def embed_battle(self, battle: Battle):
        state = torch.zeros((len(ObsIdx), 12))
        # For pokemon in battle:
        all_pokemon = [battle.active_pokemon] + battle.available_switches
        for ix, pokemon in enumerate(all_pokemon):
            # Divide by the number of types + 1 (for no type)
            state[ObsIdx.type1, ix] = pokemon.type_1.value / 19.0
            if pokemon.type_2 is not None:
                state[ObsIdx.type2, ix] = pokemon.type_2.value / 19.0

            if pokemon.ability is not None:
                state[ObsIdx.ability_bit0 : ObsIdx.ability_bit0 + 9, ix] = torch.tensor(
                    ABILITIES[
                        pokemon.ability.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("'", "")
                    ]
                )

            # Blissey has the maximum HP at 714
            if pokemon.current_hp is not None:
                state[ObsIdx.current_hp, ix] = pokemon.current_hp / 714.0

            state[ObsIdx.hp_ratio, ix] = pokemon.current_hp_fraction

            state[ObsIdx.base_hp, ix] = pokemon.base_stats["hp"] / 255.0  # Blissey
            state[ObsIdx.base_atk, ix] = pokemon.base_stats["atk"] / 190.0  # Mega Mewtwo X
            state[ObsIdx.base_def, ix] = pokemon.base_stats["def"] / 250.0  # Eternatus
            state[ObsIdx.base_spa, ix] = pokemon.base_stats["spa"] / 194.0  # Mega Mewtwo Y
            state[ObsIdx.base_spd, ix] = pokemon.base_stats["spd"] / 250.0  # Eternatus
            state[ObsIdx.base_spe, ix] = pokemon.base_stats["spe"] / 200.0  # Regieleki

            state[ObsIdx.boost_acc, ix] = pokemon.boosts["accuracy"] / 4.0
            state[ObsIdx.boost_eva, ix] = pokemon.boosts["evasion"] / 4.0
            state[ObsIdx.boost_atk, ix] = pokemon.boosts["atk"] / 4.0
            state[ObsIdx.boost_def, ix] = pokemon.boosts["def"] / 4.0
            state[ObsIdx.boost_spa, ix] = pokemon.boosts["spa"] / 4.0
            state[ObsIdx.boost_spd, ix] = pokemon.boosts["spd"] / 4.0
            state[ObsIdx.boost_spe, ix] = pokemon.boosts["spe"] / 4.0

            if pokemon.status is not None:
                state[ObsIdx.status, ix] = pokemon.status.value / 3.0

            all_moves = [move for move in pokemon.moves]
            if len(all_moves) > 4:
                for effect in pokemon.effects:
                    if effect.name == "DYNAMAX":
                        all_moves = all_moves[len(all_moves)-4:]
                        break
                else:
                    all_moves = all_moves[0:4]

            for move_ix, move in enumerate(all_moves):

                state[ObsIdx.move_1_accuracy + 7 * move_ix, ix] = pokemon.moves[move].accuracy
                state[ObsIdx.move_1_base_power + 7 * move_ix, ix] = pokemon.moves[move].accuracy
                state[ObsIdx.move_1_category + 7 * move_ix, ix] = (
                    pokemon.moves[move].category.value / 3.0
                )
                state[ObsIdx.move_1_current_pp + 7 * move_ix, ix] = (
                    pokemon.moves[move].current_pp / pokemon.moves[move].max_pp
                )
                if len(pokemon.moves[move].secondary) > 0:
                    state[ObsIdx.move_1_secondary_chance + 7 * move_ix, ix] = (
                        pokemon.moves[move].secondary[0]["chance"] / 100
                    )
                    if "status" in pokemon.moves[move].secondary[0]:
                        for value in Status:
                            if pokemon.moves[move].secondary[0]["status"] == value.name.lower():
                                state[ObsIdx.move_1_secondary_status + 7 * move_ix, ix] = value.value / 7
                state[ObsIdx.move_1_type + 7 * move_ix, ix] = pokemon.moves[move].type.value / 19.0

        for ix, pokemon in enumerate(battle.opponent_team):
            # Divide by the number of types + 1 (for no type)
            state[ObsIdx.type1, ix + 6] = battle.opponent_team[pokemon].type_1.value / 19.0
            if battle.opponent_team[pokemon].type_2 is not None:
                state[ObsIdx.type2, ix + 6] = battle.opponent_team[pokemon].type_2.value / 19.0

            # Blissey has the maximum HP at 714
            if battle.opponent_team[pokemon].current_hp is not None:
                state[ObsIdx.current_hp, ix + 6] = battle.opponent_team[pokemon].current_hp / 714.0

            state[ObsIdx.hp_ratio, ix + 6] = battle.opponent_team[pokemon].current_hp_fraction

            state[ObsIdx.base_hp, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["hp"] / 255.0
            )  # Blissey
            state[ObsIdx.base_atk, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["atk"] / 190.0
            )  # Mega Mewtwo X
            state[ObsIdx.base_def, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["def"] / 250.0
            )  # Eternatus
            state[ObsIdx.base_spa, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["spa"] / 194.0
            )  # Mega Mewtwo Y
            state[ObsIdx.base_spd, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["spd"] / 250.0
            )  # Eternatus
            state[ObsIdx.base_spe, ix + 6] = (
                battle.opponent_team[pokemon].base_stats["spe"] / 200.0
            )  # Regieleki

            state[ObsIdx.boost_acc, ix + 6] = battle.opponent_team[pokemon].boosts["accuracy"] / 4.0
            state[ObsIdx.boost_eva, ix + 6] = battle.opponent_team[pokemon].boosts["evasion"] / 4.0
            state[ObsIdx.boost_atk, ix + 6] = battle.opponent_team[pokemon].boosts["atk"] / 4.0
            state[ObsIdx.boost_def, ix + 6] = battle.opponent_team[pokemon].boosts["def"] / 4.0
            state[ObsIdx.boost_spa, ix + 6] = battle.opponent_team[pokemon].boosts["spa"] / 4.0
            state[ObsIdx.boost_spd, ix + 6] = battle.opponent_team[pokemon].boosts["spd"] / 4.0
            state[ObsIdx.boost_spe, ix + 6] = battle.opponent_team[pokemon].boosts["spe"] / 4.0

            if battle.opponent_team[pokemon].status is not None:
                state[ObsIdx.status, ix + 6] = battle.opponent_team[pokemon].status.value / 3.0

            all_moves = [move for move in battle.opponent_team[pokemon].moves]
            if len(all_moves) > 4:
                for effect in battle.opponent_team[pokemon].effects:
                    if effect.name == "DYNAMAX":
                        all_moves = all_moves[len(all_moves) - 4:]
                        break
                else:
                    all_moves = all_moves[0:4]

            for move_ix, move in enumerate(all_moves):
                state[ObsIdx.move_1_accuracy + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].accuracy
                )
                state[ObsIdx.move_1_base_power + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].accuracy
                )
                state[ObsIdx.move_1_category + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].category.value / 3.0
                )
                state[ObsIdx.move_1_current_pp + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].current_pp
                    / battle.opponent_team[pokemon].moves[move].max_pp
                )

                if len(battle.opponent_team[pokemon].moves[move].secondary) > 0:
                    state[ObsIdx.move_1_secondary_chance + 7 * move_ix, ix] = (
                        battle.opponent_team[pokemon].moves[move].secondary[0]["chance"] / 100
                    )
                    if "status" in battle.opponent_team[pokemon].moves[move].secondary[0]:
                        for value in Status:
                            if (
                                battle.opponent_team[pokemon].moves[move].secondary[0]["status"]
                                == value.name.lower()
                            ):
                                state[ObsIdx.move_1_secondary_status + 7 * move_ix, ix] = value.value / 7
                state[ObsIdx.move_1_type + 7 * move_ix, ix] = (
                    battle.opponent_team[pokemon].moves[move].type.value / 19.0
                )

        return state

    def step(self, action: int) -> Tuple:
        """

        Parameters
        ----------
        action

        Returns
        -------

        """
        if self._current_battle.finished:
            raise RuntimeError("Calling step on an envionment that hasn't been reset!")
        else:
            self._actions[self._current_battle].put(action)
            observation = self._observations[self._current_battle].get()
        return (
            observation,
            self.compute_reward(self._current_battle),
            self._current_battle.finished,
            {}
        )

    def reset(self):
        for _ in range(self.MAX_BATTLE_SWITCH_RETRY):
            battles = dict(self._actions.items())
            battles = [b for b in battles if not b.finished]
            if battles:
                self._current_battle = battles[0]
                observation = self._observations[self._curret_battle].get()
                return observation
            else:
                raise EnvironmentError("Use %s has no active battle." % self.username)


    @classmethod
    def from_args(cls):

    def close(self):

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1,
                                            victory_value=30)

    @property
    def action_space(self) -> List:
        """The action space for gen 7 single battles.

        The conversion to moves is done as follows:

            0 <= action < 4:
                The actionth available move in battle.available_moves is executed.
            4 <= action < 8:
                The action - 4th available move in battle.available_moves is executed,
                with z-move.
            8 <= action < 12:
                The action - 8th available move in battle.available_moves is executed,
                with mega-evolution.
            12 <= action < 16:
                The action - 12th available move in battle.available_moves is executed,
                while dynamaxing.
            16 <= action < 22
                The action - 16th available switch in battle.available_switches is
                executed.
        """
        return self._ACTION_SPACE