from typing import Union, Tuple

import torch
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.dqn.utils import ReplayMemory
from champion_league.utils.rollout import Rollout

GAMMA = 0.99

class ActorCriticAgent(Agent):
    def __init__(
        self,
        args
    ):
        super(ActorCriticAgent, self).__init__(args)
        self._args = args
        self._nb_actions = args.nb_actions
        self._batch_size = args.batch_size
        self._device = args.device
        self.memory = Rollout(args.memory_len)

    def learn_step(self, state, entropy_term):
        rollout = self.memory.read()
        total_reward = torch.sum(rollout.rewards)
        game_length = len(rollout)

        q_val, _ = self.forward(state)
        q_val = q_val.detach()

        q_vals = torch.zeros_like(rollout.rewards)
        for t in reversed(range(len(rollout.rewards))):
            q_val = rollout.rewards[t] + GAMMA * q_val
            q_vals[t] = q_val

        advantage = q_vals - rollout.values

        actor_loss = (-1 * rollout.log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    def step(self, battle: Union[AbstractBattle, Battle]):
        pass

    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]: