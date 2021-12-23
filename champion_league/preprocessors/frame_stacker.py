from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle

from champion_league.preprocessors.base_preprocessor import Preprocessor
from champion_league.utils.directory_utils import DotDict


class FrameStacker(Preprocessor):
    def __init__(self, sub_processor: Preprocessor, sequence_len: int, device: int):
        self.processor = sub_processor
        self.sequence_len = sequence_len
        self.frame_idx = 0
        self.device = device
        self.prev_frames = {}
        self._output_shape = {}
        for k, v in self.processor.output_shape.items():
            try:
                self._output_shape[k] = (v[0] * sequence_len, v[1] + 1)
            except TypeError:
                self._output_shape[k] = v * sequence_len

    @property
    def output_shape(self) -> Dict[str, Tuple[int, int]]:
        return self._output_shape

    def embed_battle(
        self, battle: Battle, reset: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        current_frame = self.processor.embed_battle(battle)
        if reset:
            self.reset()

        if battle.battle_tag not in self.prev_frames:
            self.prev_frames[battle.battle_tag] = {}
            for k, v in current_frame.items():
                self.prev_frames[battle.battle_tag][k] = [
                    torch.zeros_like(v).squeeze(0) for _ in range(self.sequence_len - 1)
                ]

        full_embedding = {
            k: torch.zeros(v).to(f"cuda:{self.device}")
            for k, v in self._output_shape.items()
        }

        for k, v in current_frame.items():
            if len(v.shape) == 2:
                # Scalar value (abilities, items, etc.)
                full_embedding[k] = (
                    torch.cat(
                        (*self.prev_frames[battle.battle_tag][k], v.squeeze(0)),
                        dim=-1,
                    )
                    .unsqueeze(0)
                    .type(v.dtype)
                )
            else:
                embedding = torch.cat(
                    (*self.prev_frames[battle.battle_tag][k], v.squeeze(0)),
                    dim=0,
                )
                time_info = torch.ones(self._output_shape[k][0]).to(
                    f"cuda:{self.device}"
                )
                for i in range(self.sequence_len):
                    idx1 = i * v.shape[1]
                    idx2 = (i + 1) * v.shape[1]
                    time_info[idx1:idx2] = i / max(self.sequence_len - 1, 1)

                full_embedding[k] = (
                    torch.cat((time_info.unsqueeze(-1), embedding), dim=-1)
                    .type(v.dtype)
                    .unsqueeze(0)
                )

        for k, v in current_frame.items():
            self.prev_frames[battle.battle_tag][k] = self.prev_frames[
                battle.battle_tag
            ][k][1:] + [v.squeeze(0)]
        return full_embedding

    @classmethod
    def from_args(cls, args: DotDict) -> "FrameStacker":
        from champion_league.preprocessors import build_preprocessor_from_args

        sub_processor_args = DotDict({k: v for k, v in args.items()})
        sub_processor_args.preprocessor = args.subprocessor
        sub_processor = build_preprocessor_from_args(sub_processor_args)

        return FrameStacker(sub_processor, args.sequence_len, args.device)

    def reset(self) -> None:
        self.prev_frames = {}
