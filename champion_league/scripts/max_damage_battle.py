# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.random_player import RandomPlayer

from champion_league.agent.scripted.max_damage_player import MaxDamagePlayer


async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen8randombattle", max_concurrent_battles=2
    )
    max_damage_player = MaxDamagePlayer(None)

    # Now, let's evaluate our player
    await max_damage_player.battle_against(random_player, n_battles=10_000_000)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (max_damage_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
