import time
from typing import Any
from typing import Dict

from champion_league.config import parse_args
from champion_league.scripts.imitation_learning import imitation_learning
from champion_league.scripts.league_play import league_play
from champion_league.utils.agent_utils import build_network_and_preproc
from champion_league.utils.directory_utils import get_most_recent_epoch
from champion_league.utils.poke_path import PokePath


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
    league_path = PokePath(args["logdir"], args["tag"])
    network, preprocessor = build_network_and_preproc(args)

    starting_epoch = 0
    if args["resume"]:
        network.resume(league_path.agent)
        starting_epoch = get_most_recent_epoch(league_path.agent)
    elif args["imitate"]:
        imitation_learning(
            preprocessor=preprocessor,
            network=network.train(),
            league_path=league_path,
            args=args,
        )

    league_play(
        preprocessor=preprocessor,
        network=network.eval(),
        league_path=league_path,
        args=args,
        starting_epoch=starting_epoch,
    )


if __name__ == "__main__":
    start_time = time.time()
    combination_training(parse_args(__file__))
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds!")
