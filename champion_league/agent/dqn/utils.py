import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from champion_league.env.base.obs_idx import ObsIdx

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class CNN(nn.Module):
    def __init__(self, nb_actions, in_shape):
        super().__init__()

        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_shape[0],
                in_shape[0],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for i in range(4)
        )

        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(in_shape[0]) for _ in range(4)]
        )

        relu_gain = nn.init.calculate_gain("relu")
        for i in range(4):
            self.convs[i].weight.data.mul_(relu_gain)

        self.linear1 = nn.Linear(6*(in_shape[1] // (2**4) + 1), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, nb_actions)

    def forward(self, x):
        for norm, conv in zip(self.norms, self.convs):
            x = F.relu(norm(conv(x)))

        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.softmax(self.linear3(x))

        return x


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super(DQN, self).__init__()
        # Input should be (1, 10)
        self.linear1 = nn.Linear(len(ObsIdx) * 12, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, nb_actions)
        self.bn3 = nn.BatchNorm1d(nb_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = F.relu(self.bn3(self.linear3(x)))

        return F.softmax(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.full = False

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        if len(self.memory) == self.capacity and not self.full:
            print("At capacity!\n\n\n\n")
            self.full = True
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH_SIZE = 128
GAMMA = 0.999

steps_done = 0


def greedy_policy(state, model, nb_actions, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return model(state.float()).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(nb_actions)]], device=device, dtype=torch.long
        )


def sampling_policy(state: torch.Tensor, model: nn.Module):
    return torch.multinomial(model(state.float()), 1)


def optimize_model(policy_net, target_net, memory, device, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
    non_final_next_states = (
        torch.reshape(non_final_next_states, (BATCH_SIZE, -1))
        .float()
        .to(device)
    )
    state_batch = torch.cat(batch.state)
    state_batch = torch.reshape(state_batch, (BATCH_SIZE, -1)).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch.float()).gather(
        1, action_batch
    )

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
