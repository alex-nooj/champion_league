from typing import Optional

import torch
from adept.utils.util import DotDict

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.dqn.utils import Transition


class PPOAgent(Agent):
    def __init__(
            self,
            args: DotDict,
            device: Optional[int] = None
                 ):
        super().__init__(args)
        self._args = args

    def learn_step(self, next_state):
        # Read the experience replay
        transitions = self.memory.sample(self._batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self._device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        non_final_next_states = (
            torch.reshape(non_final_next_states, (self._batch_size, -1)).float()
        )

        state_batch = torch.cat(batch.state)
        state_batch = torch.reshape(state_batch, (self._batch_size, -1)).float().to(self._device)

        action_batch = torch.cat(batch.action).to(self._device)

        reward_batch = torch.cat(batch.reward).to(self._device)

        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        next_state_values = torch.zeros(self._batch_size, device=self._device)

        # Estimate the values of the next state
        with torch.no_grad():
            pred = self.network(next_state)
            last_values = pred["critic"].squeeze(-1).data

        # Calculate the nstep reward
        gae = 0.0
        next_values = last_values
        gae_returns = []
        for i in reversed(range(rollout_len)):

