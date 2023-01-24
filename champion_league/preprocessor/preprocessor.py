import abc

import gym
from poke_env.environment import AbstractBattle
from poke_env.player import ObservationType


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        ...

    @property
    @abc.abstractmethod
    def obs_space(self) -> gym.Space:
        ...
