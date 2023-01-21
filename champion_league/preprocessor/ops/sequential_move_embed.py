import typing
from enum import auto
from enum import IntEnum

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.weather import Weather

from champion_league.preprocessor.ops.base_op import Op
from champion_league.utils.damage_helpers import calc_move_damage


class MoveIdx(IntEnum):
    dmg_1 = 0
    dmg_2 = auto()
    dmg_3 = auto()
    dmg_4 = auto()
    dmg_5 = auto()
    dmg_6 = auto()
    crit_chance = auto()  # Max 6
    acc = auto()
    drain = auto()
    heal = auto()
    pp_ratio = auto()
    recoil = auto()
    brn = auto()
    frz = auto()
    par = auto()
    psn = auto()
    slp = auto()
    tox = auto()
    con = auto()
    usr_att = auto()
    usr_def = auto()
    usr_spa = auto()
    usr_spd = auto()
    usr_spe = auto()
    usr_acc = auto()
    usr_switch = auto()
    tgt_att = auto()
    tgt_def = auto()
    tgt_spa = auto()
    tgt_spd = auto()
    tgt_spe = auto()
    tgt_acc = auto()
    stat_chance = auto()
    tgt_switch = auto()
    tgt_trap = auto()
    flinch = auto()
    prevents_sound = auto()
    priority = auto()
    breaks_protect = auto()
    protects = auto()
    light_screen = auto()
    reflect = auto()
    aurora_veil = auto()
    health = auto()
    stat_reset = auto()
    encore = auto()
    clears_tgt_hazards = auto()
    clears_usr_hazards = auto()
    physical = auto()
    special = auto()
    status = auto()


secondary_effects = {}


def gather_opponent_team(battle: Battle) -> typing.List[Pokemon]:
    opponent_team = [battle.opponent_active_pokemon]
    for mon in battle._teampreview_opponent_team:
        if mon.species not in [m.species for m in opponent_team]:
            opponent_team.append(mon)
    return opponent_team


class EmbedMoves(Op):
    def __init__(self, in_shape: typing.Tuple[int], ally: bool):
        super().__init__(in_shape)
        self.ally = ally
        self._out_shape = (in_shape[0] + 4 * len(MoveIdx),)

    def preprocess(self, battle: Battle, state: torch.Tensor) -> torch.Tensor:
        ret_tensor = torch.zeros(self._out_shape)
        ret_tensor[: self._in_shape[0]] = state

        if self.ally:
            usr = battle.active_pokemon
            tgts = gather_opponent_team(battle)
            side_conditions = [k for k in battle.opponent_side_conditions]
        else:
            usr = battle.opponent_active_pokemon
            tgts = list(battle.team.values())
            side_conditions = [k for k in battle.side_conditions]
        weather = [k for k in battle.weather]
        moves = []
        for ix, move in enumerate(usr.moves.values()):
            moves.append(
                self._embed_move(
                    move,
                    usr,
                    tgts,
                    side_conditions,
                    weather[0] if len(weather) > 0 else None,
                )
            )
        moves = torch.stack(moves).view(-1)

        ret_tensor[self._in_shape[0] : self._in_shape[0] + moves.shape[0]] = moves
        return ret_tensor

    def _embed_move(
        self,
        move: Move,
        usr: Pokemon,
        tgts: typing.List[Pokemon],
        side_conditions: typing.List[SideCondition],
        weather: typing.Union[Weather, None],
    ) -> torch.Tensor:
        move_tensor = torch.zeros(len(MoveIdx))
        for ix, tgt in enumerate(tgts):
            move_tensor[MoveIdx.dmg_1 + ix] = calc_move_damage(
                move=move,
                tgt=tgt,
                usr=usr,
                weather=weather,
                side_conditions=side_conditions,
            )
        move_tensor[MoveIdx.acc] = move.accuracy
        move_tensor[MoveIdx.drain] = move.drain
        move_tensor[MoveIdx.heal] = move.heal
        move_tensor[MoveIdx.pp_ratio] = max((move.current_pp, move.max_pp, 0))
        move_tensor[MoveIdx.recoil] = move.recoil
        for key, value in secondary_effects[move.id].items():
            move_tensor[key] = value

        move_tensor[MoveIdx.physical] = float(move.category == MoveCategory.PHYSICAL)
        move_tensor[MoveIdx.special] = float(move.category == MoveCategory.SPECIAL)
        move_tensor[MoveIdx.status] = float(move.category == MoveCategory.STATUS)

        return move_tensor

    @property
    def output_shape(self) -> typing.Tuple[int]:
        return self._out_shape
