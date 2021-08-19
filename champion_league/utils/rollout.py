from collections import namedtuple
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import torch
from adept.utils.util import DotDict

Readout = namedtuple(
    "Readout",
    [
        "states",
        "next_states",
        "actions",
        "values",
        "rewards",
        "terminals",
        "log_probs",
        "hx",
        "cx",
        "step_id",
    ],
)


class Rollout:
    def __init__(self, rollout_len: int, batch_size: int):
        """

        Parameters
        ----------
        rollout_len: int
            Length of the rollout
        """
        self.rollout_len = rollout_len
        self._batch_size = batch_size
        self.states = []
        self.next_states = []
        self.rewards = []
        self.terminals = []
        self.log_probs = []
        self.hx = []
        self.cx = []
        self.actions = []
        self.values = []
        self.step_ids = []
        self.curr_ix = 0

    def push(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        terminal: bool,
        log_probs: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
        step_nb: int,
    ) -> None:
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.log_probs.append(log_probs)
        self.hx.append(hx)
        self.cx.append(cx)
        self.actions.append(action)
        self.values.append(value)
        self.step_ids.append(step_nb)

        self.curr_ix += 1

    def clear(self) -> None:
        self.states = []
        self.next_states = []
        self.rewards = []
        self.terminals = []
        self.log_probs = []
        self.hx = []
        self.cx = []
        self.actions = []
        self.values = []
        self.step_ids = []

        self.curr_ix = 0

    def read(self) -> Readout:
        return Readout(
            states=torch.stack(self.states).view(
                self.rollout_len, self._batch_size, -1
            ),
            next_states=torch.stack(self.next_states).view(
                self.rollout_len, self._batch_size, -1
            ),
            actions=torch.stack(self.actions).view(self.rollout_len, self._batch_size),
            values=torch.stack(self.values).view(self.rollout_len, self._batch_size),
            rewards=torch.tensor(
                self.rewards, dtype=torch.float, device=self.states[0].device
            ).view(self.rollout_len, self._batch_size),
            terminals=torch.tensor(self.terminals, device=self.states[0].device).view(
                self.rollout_len, self._batch_size
            ),
            log_probs=torch.stack(self.log_probs).view(
                [self.rollout_len, self._batch_size] + list(self.log_probs[0].shape)
            ),
            hx=torch.stack(self.hx).view(
                [self.rollout_len, self._batch_size] + list(self.hx[0].shape)
            ),
            cx=torch.stack(self.cx).view(
                [self.rollout_len, self._batch_size] + list(self.cx[0].shape)
            ),
            step_id=torch.tensor(self.step_ids).view(
                self.rollout_len, self._batch_size
            ),
        )

    def __len__(self):
        return self.curr_ix
