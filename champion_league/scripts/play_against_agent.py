import asyncio
from pathlib import Path

from champion_league.config import parse_args
from champion_league.env import OpponentPlayer
from champion_league.utils.server_configuration import DockerServerConfiguration


def agent_battle(
    agent_dir: str,
    opponent_dir: str,
    device: int,
    nb_battles: int,
) -> None:
    """Method for having two agents play against each other and printing out the number of wins.

    Parameters
    ----------
    agent_dir: str
        The path to the agent's weights file.
    opponent_dir: str
        The path to the opponents weights file.
    device: int
        The device to load the agents onto.
    nb_battles: int
        How many battles the agents should play.

    Returns
    -------
    None
    """
    player1 = OpponentPlayer.from_path(
        path=Path(agent_dir),
        device=device,
        max_concurrent_battles=nb_battles,
        server_configuration=DockerServerConfiguration,
    )

    player2 = OpponentPlayer.from_path(
        path=Path(opponent_dir),
        device=device,
        max_concurrent_battles=nb_battles,
        server_configuration=DockerServerConfiguration,
    )

    asyncio.get_event_loop().run_until_complete(
        player1.battle_against(player2, n_battles=nb_battles)
    )

    print(player1.n_won_battles)


if __name__ == "__main__":
    agent_battle(**parse_args())
