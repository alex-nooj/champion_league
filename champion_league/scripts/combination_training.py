import asyncio
import os
import time
from typing import Dict

import torch
from adept.utils.util import DotDict

from champion_league.agent.opponent.league_player import LeaguePlayer
from champion_league.agent.ppo import PPOAgent
from champion_league.env.rl_player import RLPlayer
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.matchmaking.skill_tracker import SkillTracker
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.scripts.imitation_learning import imitation_learning
from champion_league.scripts.league_play import league_play


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
    start_time = time.time()
    preprocessor = build_preprocessor_from_args(multi_args["league"])

    multi_args["imitation"].in_shape = preprocessor.output_shape
    multi_args["league"].in_shape = preprocessor.output_shape

    network = build_network_from_args(multi_args["imitation"])

    args = multi_args["imitation"]
    if multi_args["imitate"]:
        network = imitation_learning(
            dataset=args.dataset,
            split_ratio=args.split_ratio,
            device=args.device,
            batch_size=args.batch_size,
            nb_epochs=args.nb_epochs,
            lr=args.lr,
            network=network,
            logdir=os.path.join(args.logdir, "challengers"),
            tag=args.tag,
            patience=args.patience,
        )
    elif multi_args["resume"]:
        epochs = [
            epoch
            for epoch in os.listdir(os.path.join(args.logdir, "challengers", args.tag))
            if os.path.isdir(os.path.join(args.logdir, "challengers", args.tag, epoch))
        ]
        if len(epochs) > 0:
            rl_epochs = [
                int(epoch.rsplit("_")[-1]) for epoch in epochs if args.tag in epoch
            ]
            model_dir = (
                f"{args.tag}_{max(rl_epochs):05d}" if len(rl_epochs) > 0 else "sl"
            )
            model_file = "network.pt" if len(rl_epochs) > 0 else "best_model.pt"

            network.load_state_dict(
                torch.load(
                    os.path.join(
                        args.logdir, "challengers", args.tag, model_dir, model_file
                    ),
                    map_location=lambda storage, loc: storage,
                )
            )
    elif os.path.isdir(
        os.path.join(args.logdir, "challengers", args.tag, "sl")
    ) and "best_model.pt" in os.listdir(
        os.path.join(args.logdir, "challengers", args.tag, "sl")
    ):
        network.load_state_dict(
            torch.load(
                os.path.join(
                    args.logdir, "challengers", args.tag, "sl", "best_model.pt"
                ),
                map_location=lambda storage, loc: storage,
            )
        )
    args = DotDict(multi_args["league"])
    network = network.eval()

    args.in_shape = list(args.in_shape)

    matchmaker = MatchMaker(
        args.self_play_prob, args.league_play_prob, args.logdir, args.tag
    )

    agent = PPOAgent(
        device=args.device,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=os.path.join(args.logdir, "challengers"),
        tag=args.tag,
    )

    agent.save_args(args)
    agent.save_model(agent.network, 0, args)

    skilltracker = SkillTracker.from_args(args)

    league_play(
        args.battle_format,
        preprocessor,
        args.sample_moves,
        agent,
        matchmaker,
        skilltracker,
        args.nb_steps,
        args.epoch_len,
        args.batch_size,
        args,
        args.logdir,
        args.rollout_len,
    )

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds!")


if __name__ == "__main__":
    main(parse_multi_args())
