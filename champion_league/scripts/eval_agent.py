# -*- coding: utf-8 -*-
import asyncio
import time

from adept.utils.util import DotDict
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.parse_args import parse_args
from champion_league.utils.resume import resume


async def main(args: DotDict):
    start = time.time()

    args, network, _ = resume(args)
    network = network.eval()
    preprocessor = build_preprocessor_from_args(args)

    env_player = RLOpponent(
        network=network,
        preprocessor=preprocessor,
        sample_moves=False,
        battle_format="gen8randombattle",
        max_concurrent_battles=10,
        player_configuration=PlayerConfiguration(args.tag, "test")
    )

    # We create two players.
    enemy_player = SimpleHeuristicsPlayer(
        battle_format="gen8randombattle",
        max_concurrent_battles=2
    )

    await env_player.battle_against(enemy_player, n_battles=args.n_battles)

    print(
        "Agent won %d / 100 battles [this took %f seconds]"
        % (env_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(parse_args()))
