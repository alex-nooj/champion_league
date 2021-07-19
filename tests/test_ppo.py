from collections import OrderedDict

from typing import Dict

from adept.utils.util import DotDict
import gym

from champion_league.agent.ppo.ppo_agent import PPOAgent
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SimpleEnv:
    def __init__(self, map_size: int):
        self.action_space = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.map_size = map_size
        self.timer = 0
        self.goal = None
        self.curr_position = None

    def reset(self):
        self.goal = np.random.randint(low=0, high=self.map_size, size=2)
        self.curr_position = np.random.randint(low=0, high=self.map_size, size=2)
        self.timer = 0
        state = [self.curr_position[0], self.curr_position[1], self.goal[0], self.goal[1]]
        return torch.tensor([i / self.map_size for i in state])

    def step(self, action: int):
        self.curr_position[0] += self.action_space[action][0]
        self.curr_position[1] += self.action_space[action][1]

        self.curr_position[0] = max(min(self.curr_position[0], self.map_size - 1), 0)
        self.curr_position[1] = max(min(self.curr_position[1], self.map_size - 1), 0)

        self.timer += 1

        if self.curr_position[0] == self.goal[0] and self.curr_position[1] == self.goal[1]:
            is_done = True
            reward = 1
        elif self.timer == (self.map_size * self.map_size):
            is_done = True
            reward = -1
        else:
            is_done = False
            reward = 0

        state = [self.curr_position[0], self.curr_position[1], self.goal[0], self.goal[1]]

        return (
            torch.tensor([i / self.map_size for i in state]),
            reward,
            is_done,
        )


class LinearNetwork(nn.Module):
    def __init__(self, nb_actions, input_shape):
        super(LinearNetwork, self).__init__()
        self.linears = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_shape, 16, bias=False)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(16, 16, bias=False)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(16, 16, bias=False)),
                    ("relu3", nn.ReLU()),
                ]
            )
        )
        self.outputs = nn.ModuleDict(
            {
                "critic": nn.Linear(16, 1, bias=True),
                "action": nn.Sequential(
                    OrderedDict(
                        [
                            ("linear_out", nn.Linear(16, nb_actions, bias=True)),
                            ("softmax", nn.Softmax(dim=-1)),
                        ]
                    )
                ),
            }
        )

    def forward(self, x, internals):
        x = self.linears(x)
        outputs = {key: self.outputs[key](x) for key in self.outputs}
        return outputs, {k: torch.zeros_like(internals[k]) for k in internals}

    def new_internals(self) -> Dict[str, torch.Tensor]:
        return {
            "hx": torch.zeros(1),
            "cx": torch.zeros(1),
        }


def train_loop(agent, env, nb_steps, network, device, learning_rate, rollout_len, batch_size):
    agent.reset()
    state = env.reset()

    next_states = []
    learn_internals = {"hx": [], "cx": []}
    internals = {k: r.view(1, -1).to(device) for k, r in network.new_internals().items()}

    optimizer = torch.optim.Adam(network.parameters(), learning_rate)
    total_nb_steps = 0
    rollout_steps = 0
    nb_games = 0
    rewards = 0
    while total_nb_steps < nb_steps:
        state = torch.from_numpy(state).float().to(device).view(1, -1)

        pred, next_internals = network(state, internals)

        action = agent.choose_move(pred["action"])

        next_state, reward, done, _ = env.step(action.item())

        rewards += reward

        if rollout_steps == rollout_len:
            next_states.append(torch.from_numpy(next_state).float().to(device))
            learn_internals["hx"].append(next_internals["hx"])
            learn_internals["cx"].append(next_internals["cx"])
            rollout_steps = 0
        else:
            logit = pred["action"].view(1, -1)
            log_softmax = F.log_softmax(logit, dim=1)
            log_prob = log_softmax.gather(1, action)

            agent.memory.push(
                state=state,
                next_state=torch.from_numpy(next_state).float().to(device),
                action=action,
                value=pred["critic"],
                reward=reward,
                terminal=done,
                log_probs=log_prob,
                hx=internals["hx"],
                cx=internals["cx"],
                step_nb=total_nb_steps,
            )
            rollout_steps += 1

        state = next_state
        internals = next_internals

        if len(next_states) == batch_size and total_nb_steps > 0:
            loss = agent.learn_step(
                optimizer,
                network,
                torch.stack(next_states).to(device),
                {
                    "hx": torch.stack(learn_internals["hx"]).squeeze(1).to(device),
                    "cx": torch.stack(learn_internals["cx"]).squeeze(1).to(device),
                },
            )
            agent.log_to_tensorboard(total_nb_steps, loss=loss)
            agent.reset()
            next_states = []

        total_nb_steps += 1

        if done:
            agent.log_to_tensorboard(nb_games, reward=rewards)
            print(f"Game {nb_games}: {rewards} {state}")
            state = env.reset()
            rewards = 0
            nb_games += 1


def test_loop(network, env, device):
    state = env.reset()
    internals = {k: r.view(1, -1).to(device) for k, r in network.new_internals().items()}

    nb_games = 0
    rewards = 0
    while nb_games < 100:
        state = state.float().to(device).view(1, -1)
        pred, next_internals = network(state, internals)
        action = torch.argmax(pred["action"], dim=-1)
        state, reward, done = env.step(action.item())
        if done:
            rewards += reward
            state = env.reset()
            nb_games += 1
    print(f"Average reward: {rewards / nb_games}")


def test_ppo():
    args = DotDict(
        {"logdir": "/home/alex/Desktop/tests/", "tag": "cartpole4", "agent_type": "challengers"}
    )

    rollout_len = 20
    batch_size = 256
    discount = 0.99
    entropy_weight = 0.01
    gae_discount = 0.95
    policy_clipping = 0.2
    device = 0
    nb_rollout_epoch = 4
    rollout_minibatch_len = 32
    map_size = 10
    nb_steps = 1_000_000
    learning_rate = 0.02

    agent = PPOAgent(
        args=args,
        rollout_len=rollout_len,
        batch_size=batch_size,
        discount=discount,
        entropy_weight=entropy_weight,
        gae_discount=gae_discount,
        policy_clipping=policy_clipping,
        device=device,
        nb_rollout_epoch=nb_rollout_epoch,
        rollout_minibatch_len=rollout_minibatch_len,
    )

    # env = SimpleEnv(map_size)
    env = gym.make("CartPole-v0")
    network = LinearNetwork(2, 4)
    network = network.to(device)
    network.eval()
    agent.save_model(network, 0)

    train_loop(agent, env, nb_steps, network, device, learning_rate, rollout_len, batch_size)
    test_loop(network, env, device)


if __name__ == "__main__":
    test_ppo()
