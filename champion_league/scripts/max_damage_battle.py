# -*- coding: utf-8 -*-
import asyncio
from typing import Any

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.teambuilder.teambuilder import Teambuilder

from champion_league.teams.team_builder import load_team_from_file
from champion_league.utils.random_team_generator import generate_random_team
from champion_league.utils.server_configuration import DockerServerConfiguration


class RandomTeamFromPool(Teambuilder):
    def yield_team(self):
        team = load_team_from_file(
            "/home/anewgent/Documents/pokemon_trainers/challengers/quick_test/"
        )
        print(team)
        parsed_team = self.parse_showdown_team(team)
        joined_team = self.join_team(parsed_team)
        return joined_team


custom_builder = RandomTeamFromPool()


class RandomPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle: AbstractBattle) -> Any:
        return battle

    def choose_move(self, battle) -> BattleOrder:
        return self.choose_random_move(battle)


def battle_run(player):
    done = False
    _ = player.reset()
    while not done:
        action = np.random.randint(low=0, high=11)
        print(action)
        _, _, done, _ = player.step(action)
    player.compete_current_battle()


def main():
    # We create two players
    player_1 = RandomPlayer(
        battle_format="gen8ou",
        team=custom_builder,
        server_configuration=DockerServerConfiguration,
    )

    player_2 = RandomPlayer(
        battle_format="gen8ou",
        team=custom_builder,
        server_configuration=DockerServerConfiguration,
    )

    # await player_1.battle_against(player_2, n_battles=100)
    player_1.play_against(
        env_algorithm=battle_run,
        opponent=player_2,
        env_algorithm_kwargs={},
    )


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    main()
