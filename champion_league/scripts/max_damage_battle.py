# -*- coding: utf-8 -*-
import asyncio

from champion_league.agent.scripted.max_damage_player import MaxDamagePlayer
from champion_league.agent.scripted.random_player import RandomPlayer


async def main(player1, player2):
    # Now, let's evaluate our player
    await player1.battle_against(player2, n_battles=100)


def run_loop(player1, player2):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(player1, player2))


if __name__ == "__main__":
    import tracemalloc
    import time
    tracemalloc.start()
    # We create two players.

    for _ in range(1000):
        start_time = time.time()
        random_player = RandomPlayer(None)
        max_damage_player = MaxDamagePlayer(None)
        run_loop(max_damage_player, random_player)
        max_damage_player.reset_battles()
        random_player.reset_battles()

        del max_damage_player
        del random_player
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print(time.time() - start_time)
        print("[ Top 5 ]")
        for stat in top_stats[:5]:
            print(stat)
