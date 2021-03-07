import os
from typing import Optional

import torch
import torch.nn.functional as F
from adept.utils.util import DotDict
from torch import optim

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.dqn.utils import (
    ReplayMemory, Transition, )
from champion_league.network.linear_three_layer import LinearThreeLayer

TARGET_UPDATE = 10


class DQNAgent(Agent):
    def __init__(
            self,
            args: DotDict,
            gamma: Optional[float] = 0.999,
            device: Optional[int] = None,
            nb_train_episodes: Optional[int] = 10_000_000,
            memory_len: Optional[int] = 100_000,
            training: Optional[bool] = True
    ):
        super().__init__(args)
        self._args = args
        self._nb_actions = args.nb_actions
        self._batch_size = args.batch_size
        self._device = args.device
        self._nb_train_episodes = args.nb_train_episodes or nb_train_episodes
        self._memory_len = args.memory_len
        self._gamma = args.gamma or gamma
        self.memory = ReplayMemory(args.memory_len or memory_len)
        self._training = training

        self.network = LinearThreeLayer(self._nb_actions)
        # self.network = CNN(self._nb_actions, (6, POKEMON_LEN + MOVE_LEN * 4))
        # if os.listdir(os.path.join(args.logdir, args.tag)):
        #     self.network = self.load_model(self.network)

        # self.target_net = DQN(self._nb_actions)
        # self.target_net = CNN(self._nb_actions, (6, POKEMON_LEN + MOVE_LEN * 4))
        self.target_net = LinearThreeLayer(self._nb_actions)
        self.network = self.network.to(self._device)
        self.target_net = self.target_net.to(self._device)

        self.target_net.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters())
        self._nb_learn_steps = 0
        self.network.eval()
        self.target_net.eval()

    def learn_step(self):
        self.network.train()
        self.target_net.train()
        if len(self.memory) < self._batch_size:
            return

        transitions = self.memory.sample(self._batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self._device,
            dtype=torch.bool,
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

        state_action_values = self.network(state_batch).gather(
            1, action_batch
        )

        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states.to(self._device)).max(1)[0].detach().to(self._device)
        )

        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self._nb_learn_steps += 1
        if self._nb_learn_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.network.state_dict())

        self.network.eval()
        self.target_net.eval()

    def step(self, state: torch.Tensor):
        if self._training:
            return torch.multinomial(self.network(state.float()), 1)
        else:
            return torch.argmax(self.network(state.float()))

    def choose_move(self, state: torch.Tensor) -> torch.Tensor:
        self.target_net.eval()
        return torch.multinomial(self.target_net(state), 1)
