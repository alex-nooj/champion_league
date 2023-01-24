import typing

import gym
from poke_env import PlayerConfiguration
from poke_env import ServerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.player import ObservationType
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

from champion_league.preprocessor import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme


class PokemonEnv(Gen8EnvSinglePlayer):
    def __init__(
        self,
        opponent: Player,
        reward_scheme: RewardScheme,
        preprocessor: Preprocessor,
        player_configuration: typing.Optional[PlayerConfiguration] = None,
        *,
        avatar: typing.Optional[int] = None,
        battle_format: typing.Optional[str] = None,
        log_level: typing.Optional[int] = None,
        save_replays: typing.Union[bool, str] = False,
        server_configuration: typing.Optional[ServerConfiguration] = None,
        start_listening: bool = True,
        start_timer_on_battle_start: bool = False,
        ping_interval: typing.Optional[float] = 20.0,
        ping_timeout: typing.Optional[float] = 20.0,
        team: typing.Optional[typing.Union[str, Teambuilder]] = None,
        start_challenging: bool = True,
        use_old_gym_api: bool = True,  # False when new API is implemented in most ML libs
    ):
        self.preprocessor = preprocessor
        self.reward_scheme = reward_scheme

        super().__init__(
            opponent=opponent,
            player_configuration=player_configuration,
            avatar=avatar,
            battle_format=battle_format
            if battle_format is not None
            else self._DEFAULT_BATTLE_FORMAT,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_listening=start_listening,
            start_timer_on_battle_start=start_timer_on_battle_start,
            team=team,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            start_challenging=start_challenging,
            use_old_gym_api=use_old_gym_api,
        )

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> ObservationType:
        return sum([v for v in self.reward_scheme.compute(current_battle).values()])

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return self.preprocessor.embed_battle(battle)

    def describe_embedding(self) -> gym.Space:
        return self.preprocessor.obs_space
