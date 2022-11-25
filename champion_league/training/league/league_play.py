from typing import Optional

from poke_env.player_configuration import PlayerConfiguration
from torch import nn

from champion_league.agent import AGENTS
from champion_league.config.load_configs import save_args
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.matchmaking.league_skill_tracker import LeagueSkillTracker
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.preprocessor import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.team_builder import AgentTeamBuilder
from champion_league.training.league.league_args import LeagueArgs
from champion_league.training.league.league_epoch import league_epoch
from champion_league.training.league.utils import beating_league
from champion_league.training.league.utils import move_to_league
from champion_league.utils.directory_utils import PokePath
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


def league_play(
    preprocessor: Preprocessor,
    network: nn.Module,
    league_path: PokePath,
    args: LeagueArgs,
    epoch: Optional[int] = 0,
):
    """Main loop for training a league agent.

    Args:
        league_path: Path to the league directory
        preprocessor: The preprocessor that this agent will be using to convert Battle objects to tensors.
        network: The network that will be training.
        args: Hyperparameters used for training.
        epoch: If we're resuming, this is the epoch we're resuming from.
    """

    agent = AGENTS[args.agent](
        league_path=league_path, tag=args.tag, resume=True, **args.agent_args
    )

    step_counter = StepCounter()
    skill_tracker = LeagueSkillTracker(league_path, args.resume)
    matchmaker = MatchMaker(
        args.probs["self_play_prob"], args.probs["league_play_prob"], league_path
    )
    team_builder = AgentTeamBuilder(
        agent_path=league_path.agent, battle_format=args.battle_format
    )

    team_builder.save_team()
    player = RLPlayer(
        battle_format=args.battle_format,
        preprocessor=preprocessor,
        reward_scheme=RewardScheme(args.rewards),
        server_configuration=DockerServerConfiguration,
        team=team_builder,
        player_configuration=PlayerConfiguration(
            username=f"rltrainer", password="rltrainer1234"
        ),
    )

    opponent = LeaguePlayer(
        device=agent.device,
        network=agent.network,
        preprocessor=preprocessor,
        sample_moves=args.sample_moves,
        max_concurrent_battles=10,
        server_configuration=DockerServerConfiguration,
        team=AgentTeamBuilder(),
        training_team=team_builder,
        battle_format=args.battle_format,
        player_configuration=PlayerConfiguration(
            username=f"rlopponent", password="rlopponent1234"
        ),
    )

    for e in range(epoch, epoch + args.nb_steps // args.epoch_len):
        agent.save_model(e, network)
        save_args(agent_dir=league_path.agent, args=args.dict_args, epoch=e)
        team_builder.save_team()

        player.play_against(
            env_algorithm=league_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "matchmaker": matchmaker,
                "skill_tracker": skill_tracker,
                "epoch_len": args.epoch_len,
                "step_counter": step_counter,
                "epoch": e,
            },
        )
        skill_tracker.save_skill_ratings(e)

        if beating_league(agent):
            move_to_league(
                agent_path=league_path.agent,
                league_dir=league_path.league,
                tag=agent.tag,
                epoch=e,
            )
            break
