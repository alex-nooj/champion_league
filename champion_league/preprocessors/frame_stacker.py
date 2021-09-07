from typing import Tuple, Dict

import torch
from champion_league.utils.directory_utils import DotDict
from poke_env.environment.battle import Battle

from champion_league.preprocessors import Preprocessor


class FrameStacker(Preprocessor):
    def __init__(self, sub_processor: Preprocessor, sequence_len: int, device: int):
        self.processor = sub_processor
        self.sequence_len = sequence_len
        self.frames = torch.zeros((self.sequence_len, *self.processor.output_shape[1:]))
        self.frame_idx = 0
        self.device = device

    @property
    def output_shape(self) -> Dict[str, Tuple[int, int]]:
        return {"pokemon": (self.sequence_len, *self.processor.output_shape[1:])}

    def embed_battle(self, battle: Battle) -> Dict[str, torch.Tensor]:
        self.frames[self.frame_idx, :] = self.processor.embed_battle(battle)[0]

        frames = torch.cat(
            (self.frames[self.frame_idx :], self.frames[: self.frame_idx]), dim=0
        )
        self.frame_idx = (self.frame_idx + 1) % self.sequence_len

        return {"pokemon": frames.unsqueeze(0).to(f"cuda:{self.device}")}

    @classmethod
    def build_from_args(cls, args: DotDict) -> "FrameStacker":
        import importlib

        subprocessor_path = f"champion_league.preprocessors.{args.subprocessor}"
        build_cls = getattr(importlib.import_module(subprocessor_path), "build_from_args")
        subprocessor = build_cls(args)

        return FrameStacker(subprocessor, args.sequence_len, args.device)
