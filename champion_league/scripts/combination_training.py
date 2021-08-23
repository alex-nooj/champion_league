import os
import time
from typing import Dict

import numpy as np

from champion_league.agent.ppo import PPOAgent
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.matchmaking.skill_tracker import SkillTracker
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.scripts.imitation_learning import imitation_learning
from champion_league.scripts.league_play import league_play
from champion_league.utils.directory_utils import DotDict, get_most_recent_epoch


def parse_multi_args() -> Dict[str, DotDict]:
    from champion_league.config import CFG

    default_args = CFG.default
    multi_args = {}
    for dict_name, arg_dict in CFG.items():
        if dict_name == "default" or dict_name == "imitate" or dict_name == "resume":
            continue

        for arg_name, arg in default_args.items():
            if arg_name not in dict(arg_dict):
                arg_dict[arg_name] = arg
        multi_args[dict_name] = arg_dict
    multi_args["imitate"] = CFG.imitate
    multi_args["resume"] = CFG.resume
    return multi_args


def main(multi_args: Dict[str, DotDict]):
    imitation_args = DotDict(multi_args["imitation"])
    league_args = DotDict(multi_args["league"])

    preprocessor = build_preprocessor_from_args(league_args)

    imitation_args.in_shape = list(preprocessor.output_shape)
    league_args.in_shape = list(preprocessor.output_shape)

    imitation_args.resume = multi_args["resume"]
    league_args.resume = multi_args["resume"]

    network = build_network_from_args(imitation_args)

    if not multi_args["resume"] and multi_args["imitate"]:
        dataset = np.load(imitation_args.dataset)
        imitation_args.in_shape = dataset["states"].shape[1:]
        network = build_network_from_args(imitation_args)

        network = imitation_learning(
            dataset=imitation_args.dataset,
            split_ratio=imitation_args.split_ratio,
            device=imitation_args.device,
            batch_size=imitation_args.batch_size,
            nb_epochs=imitation_args.nb_epochs,
            lr=imitation_args.lr,
            network=network,
            logdir=os.path.join(imitation_args.logdir, "challengers"),
            tag=imitation_args.tag,
            patience=imitation_args.patience,
        )

    network = network.eval()

    matchmaker = MatchMaker(
        league_args.self_play_prob,
        league_args.league_play_prob,
        league_args.logdir,
        league_args.tag,
    )

    agent = PPOAgent(
        device=league_args.device,
        network=network,
        lr=league_args.lr,
        entropy_weight=league_args.entropy_weight,
        clip=league_args.clip,
        logdir=os.path.join(league_args.logdir, "challengers"),
        tag=league_args.tag,
        resume=league_args.resume,
    )

    skilltracker = SkillTracker.from_args(league_args)

    if multi_args["resume"]:
        try:
            starting_epoch = get_most_recent_epoch(
                os.path.join(
                    league_args.logdir, "challengers", league_args.tag
                )
            )
        except ValueError:
            starting_epoch = 0
    else:
        starting_epoch = 0

    league_play(
        battle_format=league_args.battle_format,
        preprocessor=preprocessor,
        sample_moves=league_args.sample_moves,
        agent=agent,
        matchmaker=matchmaker,
        skilltracker=skilltracker,
        nb_steps=league_args.nb_steps,
        epoch_len=league_args.epoch_len,
        batch_size=league_args.batch_size,
        args=league_args,
        logdir=league_args.logdir,
        rollout_len=league_args.rollout_len,
        starting_epoch=starting_epoch
    )


if __name__ == "__main__":
    start_time = time.time()
    main(parse_multi_args())
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds!")
