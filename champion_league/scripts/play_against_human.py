import asyncio

from champion_league.agent.opponent.opponent_player import OpponentPlayer
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.parse_args import parse_args


async def main(args: DotDict):
    env_player = OpponentPlayer.from_path(
        get_save_dir(args.league_dir, args.tag, args.epoch), args.device
    )

    await env_player.send_challenges("anewgent", 1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(parse_args()))
