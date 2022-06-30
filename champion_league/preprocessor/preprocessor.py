from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from torch import Tensor

from champion_league.config.load_configs import get_default_args
from champion_league.preprocessor.ops import OPS


class Preprocessor:
    def __init__(
        self,
        device: int,
        ops: Optional[Dict[str, List[str]]] = None,
        op_args: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ):
        cfg = get_default_args(__file__)
        ops = ops or cfg.ops
        op_args = op_args or cfg.op_args
        self.device = device
        self._ops = {}
        self._out_shapes = {}
        for output_head in ops:
            self._ops[output_head] = []
            in_shape = (0,)
            for op_name in ops[output_head]:
                if output_head in op_args and op_name in op_args[output_head]:
                    op_arg = op_args[output_head][op_name]
                else:
                    op_arg = {}
                op = OPS[op_name](in_shape=in_shape, **op_arg)
                in_shape = op.output_shape
                self._ops[output_head].append(op)
            self._out_shapes[output_head] = in_shape

    def embed_battle(self, battle: Battle) -> Dict[str, Tensor]:
        state = {}
        for output_head in self._ops:
            out_state = torch.zeros(1)
            for op in self._ops[output_head]:
                out_state = op.preprocess(battle, out_state)
            state[output_head] = out_state.to(self.device).float()
        return state

    @property
    def output_shape(self) -> Dict[str, Tuple[int, ...]]:
        return self._out_shapes

    def reset(self):
        pass
