from pathlib import Path
from typing import Optional

import numpy as np
from torch import nn

from champion_league.agent.ppo import PPOAgent
from champion_league.config.load_configs import save_args
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.preprocessor import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.agent_team_builder import AgentTeamBuilder
from champion_league.training.agent.agent_epoch import agent_epoch
from champion_league.training.agent.agent_matchmaker import AgentMatchMaker
from champion_league.training.agent.agent_play_args import AgentPlayArgs
from champion_league.utils.directory_utils import PokePath
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


def agent_play(
    preprocessor: Preprocessor,
    network: nn.Module,
    team_builder: AgentTeamBuilder,
    league_path: PokePath,
    args: AgentPlayArgs,
    epoch: Optional[int] = 0,
) -> int:
    agent = PPOAgent(
        league_path=league_path,
        tag=args.tag,
        resume=True,
        device=args.device,
        network=network,
        **args.agent_args,
    )

    step_counter = StepCounter()

    starting_epoch = epoch
    final_epoch = epoch

    opponents = [Path(o) for o in args.opponents]
    for opponent_path in opponents:
        for e in range(
            starting_epoch, args.nb_steps // args.epoch_len + starting_epoch
        ):
            save_args(agent_dir=league_path.agent, args=args.dict_args, epoch=e)
            print(f"\nAgent {opponent_path.stem} - Epoch {e:3d}\n")

            opponent = LeaguePlayer(
                device=args.device,
                matchmaker=AgentMatchMaker(opponent_path),
                sample_moves=False,
                battle_format=args.battle_format,
            )

            player = RLPlayer(
                battle_format=args.battle_format,
                preprocessor=preprocessor,
                reward_scheme=RewardScheme(args.rewards),
                server_configuration=DockerServerConfiguration,
                team=team_builder,
            )

            player.play_against(
                env_algorithm=agent_epoch,
                opponent=opponent,
                env_algorithm_kwargs={
                    "agent": agent,
                    "opponent": opponent,
                    "epoch_len": args.epoch_len,
                    "step_counter": step_counter,
                    "epoch": e,
                },
            )
            final_epoch = e

            if np.mean(agent.win_rates[opponent.tag]) > 0.6:
                break
        starting_epoch = final_epoch
    return final_epoch
