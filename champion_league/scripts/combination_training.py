import os
import time
from typing import Dict

import numpy as np
from omegaconf import DictConfig

from champion_league.agent.imitation.imitation_agent import ImitationAgent
from champion_league.agent.ppo import PPOAgent
from champion_league.matchmaking.league_skill_tracker import LeagueSkillTracker
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.scripts.imitation_learning import imitation_learning
from champion_league.scripts.league_play import league_play
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_most_recent_epoch


def parse_multi_args() -> Dict[str, DotDict]:
    from champion_league.config import CFG

    multi_args = {}
    arg_dicts = {"imitation": CFG.imitation, "league": CFG.league}
    for arg_dict in arg_dicts:
        multi_args[arg_dict] = {**CFG.default, **arg_dicts[arg_dict]}
        for arg_name, arg_value in multi_args[arg_dict].items():
            if type(arg_value) == DictConfig:
                multi_args[arg_dict][arg_name] = DotDict(multi_args[arg_dict][arg_name])

    multi_args["imitate"] = CFG.imitate
    multi_args["resume"] = CFG.resume
    return multi_args


def main(multi_args: Dict[str, DotDict]):
    imitation_args = DotDict(multi_args["imitation"])
    league_args = DotDict(multi_args["league"])

    preprocessor = build_preprocessor_from_args(league_args)

    imitation_args.in_shape = preprocessor.output_shape
    league_args.in_shape = preprocessor.output_shape

    imitation_args.resume = multi_args["resume"]
    league_args.resume = multi_args["resume"]

    if not multi_args["resume"] and multi_args["imitate"]:
        imitation_network = build_network_from_args(imitation_args)
        imitation_learning(
            preprocessor=preprocessor,
            network=imitation_network,
            args=imitation_args,
        )
        league_network = build_network_from_args(league_args)
        league_network.load_state_dict(imitation_network.state_dict())
        starting_epoch = 0
    elif multi_args["resume"]:
        league_network = build_network_from_args(league_args)
        try:
            starting_epoch = get_most_recent_epoch(
                os.path.join(league_args.logdir, "challengers", league_args.tag)
            )
        except ValueError:
            starting_epoch = 0
    else:
        league_network = build_network_from_args(league_args)
        starting_epoch = 0

    league_play(
        preprocessor=preprocessor,
        network=league_network.eval(),
        args=league_args,
        starting_epoch=starting_epoch,
    )


if __name__ == "__main__":
    start_time = time.time()
    main(parse_multi_args())
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds!")
