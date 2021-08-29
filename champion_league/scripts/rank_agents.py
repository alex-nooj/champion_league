import asyncio
import os
import json
from typing import List, Optional

import trueskill

from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.opponent.opponent_player import OpponentPlayer
from champion_league.utils.parse_args import parse_args


async def challenge_loop(
    agent_list: List[str],
    nb_battles: Optional[int] = 100,
    player_device: Optional[int] = None,
    opp_device: Optional[int] = None,
) -> None:
    while len(agent_list) > 1:
        current_agent = agent_list.pop()
        with open(os.path.join(current_agent, "trueskill.json"), "r") as fp:
            current_skill = trueskill.Rating(**json.load(fp))

        for agent in agent_list:
            with open(os.path.join(agent, "trueskill.json"), "r") as fp:
                opponent_skill = trueskill.Rating(**json.load(fp))

            current_player = OpponentPlayer.from_path(
                path=current_agent,
                device=player_device,
                max_concurrent_battles=10,
                player_configuration=PlayerConfiguration(
                    current_agent.rsplit("/")[-1], "none"
                ),
            )

            opponent_player = OpponentPlayer.from_path(
                path=agent,
                device=opp_device,
                max_concurrent_battles=10,
                player_configuration=PlayerConfiguration(agent.rsplit("/")[-1], "none"),
            )

            await current_player.battle_against(opponent_player, n_battles=nb_battles)

            for win in current_player.battle_history:
                if win:
                    current_skill, opponent_skill = trueskill.rate_1vs1(
                        current_skill, opponent_skill
                    )
                else:
                    opponent_skill, current_skill = trueskill.rate_1vs1(
                        opponent_skill, current_skill
                    )

            with open(os.path.join(agent, "trueskill.json"), "w") as fp:
                json.dump(opponent_skill, fp, indent=4)

            current_player.reset_battles()

            del current_player
            del opponent_player

        with open(os.path.join(current_agent, "trueskill.json"), "w") as fp:
            json.dump(current_skill, fp, indent=4)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(challenge_loop(**parse_args()))
