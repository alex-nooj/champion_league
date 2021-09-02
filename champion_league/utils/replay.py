import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


def cumulative_sum(array: Union[List, np.ndarray], gamma: Optional[float] = 1.0) -> List[float]:
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma: Optional[float] = 0.99, lambd: Optional[float] = 0.95):
        self.observations = {}
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self,
        observation: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        value: float,
        log_probability: float,
        reward_scale: Optional[int] = 20,
    ):
        for key in observation:
            if key not in self.observations:
                self.observations[key] = [observation[key]]
            else:
                self.observations[key].append(observation[key].squeeze(0))
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value: float):
        rewards = np.asarray(self.rewards + [last_value])
        values = np.asarray(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)

        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array: List[float]) -> List[float]:
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = {}
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []
        self._length = 0

    def free_memory(self):
        del self.episodes[:]
        for key in self.observations:
            del self.observations[key][:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]
        self._length = 0

    def add_episode(self, episode: Episode):
        self._length += len(episode.rewards)
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            for key in episode.observations:
                if key not in self.observations:
                    self.observations[key] = []
                self.observations[key] += episode.observations[key]
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Dict[str, torch.Tensor], List[int], List[float], List[float], List[float]
    ]:
        return (
            {k: v[idx].squeeze() for k, v in self.observations},
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
