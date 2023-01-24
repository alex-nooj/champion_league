import typing

import torch
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.weather import Weather

from champion_league.preprocessor.ops.base_op import Op
from champion_league.preprocessor.util.move_effects import MoveIdx
from champion_league.utils.damage_helpers import calc_move_damage
from champion_league.utils.gather_opponent_team import gather_opponent_team


class EmbedMoves(Op):
    def __init__(self, in_shape: typing.Tuple[int], ally: bool):
        super().__init__(in_shape)
        self.ally = ally
        self._out_shape = (in_shape[0] + 4 * len(MoveIdx),)

    def preprocess(self, battle: Battle, state: torch.Tensor) -> torch.Tensor:
        ret_tensor = -1 * torch.ones(self._out_shape)
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
        if len(moves) > 0:
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
        move_tensor[MoveIdx.pp_ratio] = max((move.current_pp / move.max_pp, 0))
        move_tensor[MoveIdx.recoil] = move.recoil

        move_tensor[MoveIdx.physical] = float(move.category == MoveCategory.PHYSICAL)
        move_tensor[MoveIdx.special] = float(move.category == MoveCategory.SPECIAL)
        move_tensor[MoveIdx.status] = float(move.category == MoveCategory.STATUS)

        return move_tensor

    @property
    def output_shape(self) -> typing.Tuple[int]:
        return self._out_shape
