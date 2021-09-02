from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from typing import Optional


def cumulative_sum(rewards: np.ndarray, gamma: Optional[float] = 0.99) -> List[float]:
    """Function for calculating the n-step returns
    
    Parameters
    ----------
    rewards: np.ndarray
        The rewards from an episode
    gamma: Optional[float]
        The discount for the n-step returns

    Returns
    -------
    List[float]
        The n-step returns of an episode
    """
    curr = 0
    cumulative_array = []

    for a in rewards[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma: Optional[float] = 0.99, lambd: Optional[float] = 0.95):
        """Class for storing the important parts of an episode

        Parameters
        ----------
        gamma: Optional[float]
            The discount rate for the n-step returns
        lambd: Optional[float]
            The discount for the advantages
        """
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, observation: Dict[str, torch.Tensor], action: int, reward: float, value: float, log_probability: float, reward_scale: int=20
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)

        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.observations += episode.observations
        self.actions += episode.actions
        self.advantages += episode.advantages
        self.rewards += episode.rewards
        self.rewards_to_go += episode.rewards_to_go
        self.log_probabilities += episode.log_probabilities

    def build_dataset(self):
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

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
