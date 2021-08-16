from asyncio import Queue
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder


class RLPlayer(Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(4 * 2 + 6))

    def __init__(
        self,
        embed_battle: Callable,
        player_configuration: Optional[PlayerConfiguration] = None,
        *,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        server_configuration: Optional[ServerConfiguration] = None,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        team: Optional[Union[str, Teambuilder]] = None,
    ) -> None:
        """Player class that will act as the league. Whenever the game ends, call change_agent() to
        change the agent playing the game. Allows sampling of self-play, league-play, and exploiting

        Parameters
        ----------
        embed_battle: Callable
            Callable that accepts an AbstractBattle class and returns a tensor.
        player_configuration: Optional[PlayerConfiguration]
            Player configuration. If empty, defaults to an
            automatically generated username with no password. This option must be set
            if the server configuration requires authentication.
        avatar: Optional[int]
            Player avatar id. Optional.
        battle_format: Optional[str]
            Name of the battle format this player plays. Defaults to
            gen8randombattle.
        log_level: Optional[int]
            The player's logger level.
        server_configuration: Optional[ServerConfiguration]
            Server configuration. Defaults to Localhost Server Configuration
        start_timer_on_battle_start: bool
            Whether or not to start the battle timer
        start_listening: bool
            Whether to start listening to the server. Defaults to True
        team: Optional[Union[str, Teambuilder]]
            The team to use for formats requiring a team. Can be a showdown team string, a showdown
            packed team string, or a ShowdownTeam object. Defaults to None.
        """

        super().__init__(
            player_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            team=team,
        )
        self._max_concurrent_battles = 2
        self._battle_count_queue = Queue(2)
        self.embed_battle_cls = embed_battle

    def embed_battle(self, battle):
        return self.embed_battle_cls(battle=battle)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=1, hp_value=0, victory_value=0
        )

    def step(self, action: int) -> Tuple:
        obs, reward, done, _ = super().step(action)
        return obs, reward, done, {"won": 1 if self._current_battle.won else 0}

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)
