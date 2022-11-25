import asyncio
import pathlib
import typing

from champion_league.config import parse_args
from champion_league.env import OpponentPlayer
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.server_configuration import DockerServerConfiguration


async def main(args: typing.Dict[str, typing.Any]):
    env_player = OpponentPlayer(
        get_save_dir(pathlib.Path(args["league_dir"], args["tag"]), args["epoch"]),
        args["device"],
        server_configuration=DockerServerConfiguration,
    )

    await env_player.send_challenges(args["challenger"], 1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(parse_args(__file__)))
