import typing

import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Move
from poke_env.environment import MoveCategory
from poke_env.environment import Pokemon
from poke_env.environment import PokemonType
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather
from poke_env.player import ObservationType

from champion_league.preprocessor import Preprocessor
from champion_league.preprocessor.ops.embed_team import grounded
from champion_league.preprocessor.ops.embed_team import TeamIdx
from champion_league.preprocessor.util.move_effects import MoveIdx
from champion_league.preprocessor.util.normalize_stats import normalize_stats
from champion_league.utils.damage_helpers import calc_move_damage
from champion_league.utils.gather_opponent_team import gather_opponent_team


def determine_entry_hazard(
    pokemon: Pokemon,
    side_conditions: typing.List[SideCondition],
) -> float:
    dmg = 0.0
    if SideCondition.SPIKES in side_conditions and not grounded(pokemon):
        dmg += 1 / 6

    if (
        SideCondition.STEALTH_ROCK in side_conditions
        and pokemon.ability != "magicguard"
    ):
        if pokemon.damage_multiplier(PokemonType.ROCK) == 0.25:
            dmg += 0.0312
        elif pokemon.damage_multiplier(PokemonType.ROCK) == 0.5:
            dmg += 0.062
        elif pokemon.damage_multiplier(PokemonType.ROCK) == 1.0:
            dmg += 0.125
        elif pokemon.damage_multiplier(PokemonType.ROCK) == 2.0:
            dmg += 0.25
        else:
            dmg += 0.5
    return dmg


class FullPreprocessor(Preprocessor):
    def __init__(self):
        self._ix = {}
        self.team_keys = [
            "atk_stat",
            "def_stat",
            "spa_stat",
            "spd_stat",
            "spe_stat",
            "burn",
            "faint",
            "freeze",
            "paralyze",
            "poisoned",
            "sleep",
            "toxic",
            "side_condition",
            "dmg_1",
            "dmg_2",
            "dmg_3",
            "dmg_4",
            "opp_dmg_1",
            "opp_dmg_2",
            "opp_dmg_3",
            "opp_dmg_4",
            "bug",
            "dark",
            "dragon",
            "electric",
            "fairy",
            "fighting",
            "fire",
            "flying",
            "ghost",
            "grass",
            "ground",
            "ice",
            "normal",
            "poison",
            "psychic",
            "rock",
            "steel",
            "water",
        ]
        self.active_keys = [
            "hp_fraction",
            "atk_stat",
            "def_stat",
            "spa_stat",
            "spd_stat",
            "spe_stat",
            "acc_boost",
            "eva_boost",
            "bug",
            "dark",
            "dragon",
            "electric",
            "fairy",
            "fighting",
            "fire",
            "flying",
            "ghost",
            "grass",
            "ground",
            "ice",
            "normal",
            "poison",
            "psychic",
            "rock",
            "steel",
            "water",
            "burn",
            "faint",
            "freeze",
            "paralyze",
            "poisoned",
            "sleep",
            "toxic",
        ]

        self.move_keys = [
            "dmg_1",
            "dmg_2",
            "dmg_3",
            "dmg_4",
            "dmg_5",
            "dmg_6",
            "crit_chance",
            "acc",
            "drain",
            "heal",
            "pp_ratio",
            "recoil",
            "physical",
            "special",
            "status",
        ]

        full_keys = []
        for _ in range(2):
            full_keys += self.active_keys
            for _ in range(4):
                full_keys += self.move_keys
        for _ in range(5):
            full_keys += self.team_keys

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        active_pokemon = self._embed_active(battle.active_pokemon)
        usr = battle.active_pokemon
        tgts = gather_opponent_team(battle)
        side_conditions = [k for k in battle.opponent_side_conditions]
        weather = [k for k in battle.weather]
        if len(weather) > 0:
            weather = weather[0]
        else:
            weather = None

        moves = np.zeros(4 * len(self.move_keys))
        for ix, move in enumerate(battle.active_pokemon.moves.values()):
            moves[
                ix * len(self.move_keys) : (ix + 1) * len(self.move_keys)
            ] += self._embed_move(move, usr, tgts, side_conditions, weather)

        opponent_pokemon = self._embed_active(battle.opponent_active_pokemon)

        opponent_moves = np.zeros(4 * len(self.move_keys))
        for ix, move in enumerate(battle.opponent_active_pokemon.moves.values()):
            opponent_moves[
                ix * len(self.move_keys) : (ix + 1) * len(self.move_keys)
            ] += self._embed_move(
                move,
                battle.opponent_active_pokemon,
                [mon for mon in battle.team.values()],
                [k for k in battle.side_conditions],
                weather,
            )

        ally_team = [
            self._embed_non_active(
                pokemon,
                battle.opponent_active_pokemon,
                [k for k in battle.side_conditions],
                weather,
                side_conditions,
            )
            for pokemon in battle.team.values()
            if not pokemon.active
        ]

        return np.concatenate(
            (active_pokemon, moves, opponent_pokemon, opponent_moves, ally_team), axis=0
        )

    def _embed_active(self, pokemon: Pokemon) -> npt.NDArray:
        stats = normalize_stats(pokemon).numpy()

        status = np.zeros(len(Status))
        if pokemon.status is not None:
            status[pokemon.status.value - 1] = 1.0
        poke_types = np.zeros(18)
        poke_types[pokemon.type_1.value - 1] = 1.0
        if pokemon.type_2 is not None:
            poke_types[pokemon.type_2.value - 1] = 1.0

        return np.concatenate((stats, poke_types, status), axis=0)

    def _embed_move(
        self,
        move: Move,
        usr: Pokemon,
        tgts: typing.List[Pokemon],
        side_conditions: typing.List[SideCondition],
        weather: typing.Union[Weather, None],
    ) -> npt.NDArray:
        move_embedding = np.zeros(len(MoveIdx))
        for ix, tgt in enumerate(tgts):
            move_embedding[MoveIdx.dmg_1 + ix] = calc_move_damage(
                move=move,
                tgt=tgt,
                usr=usr,
                weather=weather,
                side_conditions=side_conditions,
            )
        move_embedding[MoveIdx.acc] = move.accuracy
        move_embedding[MoveIdx.drain] = move.drain
        move_embedding[MoveIdx.heal] = move.heal
        move_embedding[MoveIdx.pp_ratio] = max((move.current_pp / move.max_pp, 0))
        move_embedding[MoveIdx.recoil] = move.recoil

        move_embedding[MoveIdx.physical] = float(move.category == MoveCategory.PHYSICAL)
        move_embedding[MoveIdx.special] = float(move.category == MoveCategory.SPECIAL)
        move_embedding[MoveIdx.status] = float(move.category == MoveCategory.STATUS)

        return move_embedding

    def _embed_non_active(
        self,
        pokemon: Pokemon,
        opponent: Pokemon,
        side_conditions: typing.List[SideCondition],
        weather: typing.Union[Weather, None],
        opponent_side_conditions: typing.List[SideCondition],
    ) -> npt.NDArray:
        pokemon_embedding = np.zeros(len(self.team_keys))
        stats = normalize_stats(pokemon)
        pokemon_embedding[: len(stats)] += stats.numpy()
        if pokemon.status is not None:
            pokemon_embedding[TeamIdx.burn + pokemon.status.value - 1] = 1.0
        elif (
            SideCondition.TOXIC_SPIKES in side_conditions
            and (
                PokemonType.POISON not in pokemon.types
                or PokemonType.STEEL not in pokemon.types
            )
            and pokemon.item != "airballoon"
        ):
            pokemon_embedding[TeamIdx.poison] = 1.0

        pokemon_embedding[TeamIdx.side_condition] = determine_entry_hazard(
            pokemon, side_conditions
        )
        for m_ix, move in enumerate(pokemon.moves.values()):
            pokemon_embedding[TeamIdx.dmg_1 + m_ix] = calc_move_damage(
                move=move,
                tgt=opponent,
                usr=pokemon,
                weather=weather,
                side_conditions=side_conditions,
            )
        for m_ix, move in enumerate(opponent.moves.values()):
            pokemon_embedding[TeamIdx.dmg_1 + m_ix] = calc_move_damage(
                move=move,
                tgt=pokemon,
                usr=opponent,
                weather=weather,
                side_conditions=opponent_side_conditions,
            )

        for pokemon_type in pokemon.types:
            if pokemon_type is None:
                continue
            pokemon_embedding[TeamIdx.bug + pokemon_type.value - 1] = 1.0

        return pokemon_embedding
