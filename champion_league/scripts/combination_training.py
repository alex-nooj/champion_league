import os
import time
from typing import Any
from typing import Dict

from omegaconf import DictConfig

from champion_league.config import parse_args
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.scripts.imitation_learning import imitation_learning
from champion_league.scripts.league_play import league_play
from champion_league.utils.directory_utils import get_most_recent_epoch


def combination_training(args: Dict[str, Any]) -> None:
    """Method for performing imitation learning and league training sequentially.

    Parameters
    ----------
    args
        The arguments for imitation learning, league play, and meta commands for this script.

    Returns
    -------
    None
    """

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
    combination_training(parse_args())
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds!")
