from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch.utils.data import Dataset

from champion_league.agent.base.base_replay_buffer import ReplayBuffer
from champion_league.utils.replay import Episode
from champion_league.utils.replay import normalize_list


class PPOReplayBuffer(ReplayBuffer, Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []
        self._length = 0

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
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
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        self.advantages = normalize_list(self.advantages)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        List[Dict[str, torch.Tensor]],
        List[int],
        List[float],
        List[float],
        List[float],
    ]:
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
