# -*- coding: utf-8 -*-
import asyncio
import json
import os

import numpy as np
import torch
from adept.utils.util import DotDict
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args


def parse_args() -> DotDict:
    from champion_league.config import CFG

    args = DotDict(
        {k: tuple(v) if type(v) not in [int, float, str, bool] else v for k, v in CFG.items()}
    )

    with open(os.path.join(args.logdir, args.tag, "args.json"), "r") as fp:
        old_args = json.load(fp)
        old_args = DotDict(old_args)
    old_args.gpu_id = args.gpu_id
    old_args.epoch = args.epoch
    return old_args


def load_model(network: torch.nn.Module, logdir: str, tag: str, epoch: int) -> torch.nn.Module:
    model_file = os.path.join(logdir, tag, f"{tag}_{epoch:05d}", "network.pth")
    network.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    return network


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def teampreview(self, battle):
        mon_performance = {}

        # For each of our pokemons
        for i, mon in enumerate(battle.team.values()):
            # We store their average performance against the opponent team
            mon_performance[i] = np.mean(
                [teampreview_performance(mon, opp) for opp in battle.opponent_team.values()]
            )

        # We sort our mons by performance
        ordered_mons = sorted(mon_performance, key=lambda k: -mon_performance[k])

        # We start with the one we consider best overall
        # We use i + 1 as python indexes start from 0
        #  but showdown's indexes start from 1
        return "/team " + "".join([str(i + 1) for i in ordered_mons])


def teampreview_performance(mon_a, mon_b):
    # We evaluate the performance on mon_a against mon_b as its type advantage
    a_on_b = b_on_a = -np.inf
    for type_ in mon_a.types:
        if type_:
            a_on_b = max(a_on_b, type_.damage_multiplier(*mon_b.types))
    # We do the same for mon_b over mon_a
    for type_ in mon_b.types:
        if type_:
            b_on_a = max(b_on_a, type_.damage_multiplier(*mon_a.types))
    # Our performance metric is the different between the two
    return a_on_b - b_on_a


async def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)

    network = build_network_from_args(args).eval()

    network = load_model(network, args.logdir, args.tag, args.epoch)

    env_player = RLOpponent(
        network=network,
        preprocessor=preprocessor,
        sample_moves=False,
        battle_format="gen8randombattle",
        max_concurrent_battles=10,
        player_configuration=PlayerConfiguration("bot_username", "bot_password"),
    )

    await env_player.send_challenges("anewgent", 1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(parse_args()))
