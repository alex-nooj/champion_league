import typing
from enum import auto
from enum import IntEnum

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition

from champion_league.preprocessor.ops.base_op import Op
from champion_league.preprocessor.util.normalize_stats import normalize_stats
from champion_league.utils.damage_helpers import calc_move_damage
from champion_league.utils.gather_opponent_team import gather_opponent_team


class TeamIdx(IntEnum):
    hp_fraction = 0
    atk_stat = auto()
    def_stat = auto()
    spa_stat = auto()
    spd_stat = auto()
    spe_stat = auto()
    burn = auto()
    faint = auto()
    freeze = auto()
    paralyze = auto()
    poisoned = auto()
    sleep = auto()
    toxic = auto()
    side_condition = auto()
    dmg_1 = auto()
    dmg_2 = auto()
    dmg_3 = auto()
    dmg_4 = auto()
    opp_dmg_1 = auto()
    opp_dmg_2 = auto()
    opp_dmg_3 = auto()
    opp_dmg_4 = auto()
    bug = auto()
    dark = auto()
    dragon = auto()
    electric = auto()
    fairy = auto()
    fighting = auto()
    fire = auto()
    flying = auto()
    ghost = auto()
    grass = auto()
    ground = auto()
    ice = auto()
    normal = auto()
    poison = auto()
    psychic = auto()
    rock = auto()
    steel = auto()
    water = auto()


def grounded(pokemon: Pokemon) -> bool:
    return not (
        PokemonType.FLYING in pokemon.types
        or pokemon.ability == "levitate"
        or pokemon.item == "airballoon"
    )


class EmbedTeam(Op):
    def __init__(self, in_shape: typing.Tuple[int], ally: bool):
        super().__init__(in_shape)
        self._out_shape = (
            5,
            len(TeamIdx),
        )
        self.ally = ally

    def preprocess(self, battle: Battle, state: torch.Tensor) -> torch.Tensor:
        ret_tensor = torch.zeros((5, len(TeamIdx)))
        if self.ally:
            opponent = battle.opponent_active_pokemon
            side_conditions = list(battle.side_conditions.keys())
            opponent_side_conditions = list(battle.opponent_side_conditions.keys())
            team = list(battle.team.values())
        else:
            opponent = battle.active_pokemon
            side_conditions = list(battle.opponent_side_conditions.keys())
            opponent_side_conditions = list(battle.side_conditions.keys())
            team = gather_opponent_team(battle)

        weather = [k for k in battle.weather]
        weather = weather[0] if len(weather) > 0 else None
        ix = 0
        for pokemon in team:
            if pokemon.active:
                continue
            stats = normalize_stats(pokemon)
            ret_tensor[ix, TeamIdx.hp_fraction] = stats[0]
            ret_tensor[ix, TeamIdx.atk_stat] = stats[1]
            ret_tensor[ix, TeamIdx.def_stat] = stats[2]
            ret_tensor[ix, TeamIdx.spa_stat] = stats[3]
            ret_tensor[ix, TeamIdx.spd_stat] = stats[4]
            ret_tensor[ix, TeamIdx.spe_stat] = stats[5]

            if pokemon.status is not None:
                ret_tensor[ix, TeamIdx.burn + pokemon.status.value - 1] = 1.0
            elif (
                SideCondition.TOXIC_SPIKES in battle.side_conditions
                and (
                    PokemonType.POISON not in pokemon.types
                    or PokemonType.STEEL not in pokemon.types
                )
                and pokemon.item != "airballoon"
            ):
                ret_tensor[ix, TeamIdx.poison] = 1.0

            ret_tensor[ix, TeamIdx.side_condition] = self._determine_entry_hazard(
                battle, pokemon, side_conditions
            )
            for m_ix, move in enumerate(pokemon.moves.values()):
                ret_tensor[ix, TeamIdx.dmg_1 + m_ix] = calc_move_damage(
                    move=move,
                    tgt=opponent,
                    usr=pokemon,
                    weather=weather,
                    side_conditions=side_conditions,
                )
            for m_ix, move in enumerate(opponent.moves.values()):
                ret_tensor[ix, TeamIdx.dmg_1 + m_ix] = calc_move_damage(
                    move=move,
                    tgt=pokemon,
                    usr=opponent,
                    weather=weather,
                    side_conditions=opponent_side_conditions,
                )

            for pokemon_type in pokemon.types:
                if pokemon_type is None:
                    continue
                ret_tensor[ix, TeamIdx.bug + pokemon_type.value - 1] = 1.0
            ix += 1
            if ix == 5:
                break

        return ret_tensor

    def _determine_entry_hazard(
        self,
        battle: Battle,
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

    @property
    def output_shape(self) -> typing.Tuple[int, int]:
        return self._out_shape
