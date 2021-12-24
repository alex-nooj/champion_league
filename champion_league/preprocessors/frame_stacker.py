from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle

from champion_league.preprocessors.base_preprocessor import Preprocessor
from champion_league.utils.directory_utils import DotDict


class FrameStacker(Preprocessor):
    """Preprocessor that stacks the output from another preprocessor."""

    def __init__(self, sub_processor: Preprocessor, sequence_len: int, device: int):
        """Constructor

        Parameters
        ----------
        sub_processor
            The preprocessing scheme we'd like to use and stack.
        sequence_len
            The number of previous frames to include in the output.
        device
            The device to move the tensors to.
        """
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

    def embed_battle(
        self, battle: Battle, reset: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        """Method that calls the subprocessor on the current frame, then stacks the previous frames
        onto it, embedding the temporal data into each frame.

        Parameters
        ----------
        battle
            The Battle object (game state) to be preprocessed.
        reset
            Whether or not to reset the preprocessing.

        Returns
        -------
        Dict[str, Tensor]
            The state, preprocessed into a form that is useable by the neural network.
        """
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

    @property
    def output_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Class property describing the preprocessor's output shape.

        Returns
        -------
        Dict[str, Tuple[int, ...]]
            The output shape for each head of the preprocessor.
        """
        return self._output_shape

    @classmethod
    def from_args(cls, args: DotDict) -> "FrameStacker":
        """Constructor for building this preprocessor from arguments.

        Parameters
        ----------
        args
            The arguments to construct this preprocessor. In addition to the key `subprocessor`,
            these arguments MUST include all the arguments to build the subprocessor.

        Returns
        -------
        FrameStacker
            An instance of the FrameStacker class.
        """
        from champion_league.preprocessors import build_preprocessor_from_args

        sub_processor_args = DotDict({k: v for k, v in args.items()})
        sub_processor_args.preprocessor = args.subprocessor
        sub_processor = build_preprocessor_from_args(sub_processor_args)

        return FrameStacker(sub_processor, args.sequence_len, args.device)

    def reset(self) -> None:
        """Empties the previous frames."""
        self.prev_frames = {}
