import asyncio
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
from poke_env.player_configuration import PlayerConfiguration
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.ppo import PPOAgent
from champion_league.config import parse_args
from champion_league.config.load_configs import save_args
from champion_league.env import OpponentPlayer
from champion_league.env.league_player import LeaguePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.matchmaking.league_skill_tracker import LeagueSkillTracker
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.preprocessor import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.team_builder import AgentTeamBuilder
from champion_league.utils.agent_utils import build_network_and_preproc
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.poke_path import PokePath
from champion_league.utils.progress_bar import centered
from champion_league.utils.replay import History
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


def print_table(entries: Dict[str, float], float_precision: Optional[int] = 1) -> None:
    """Used for printing a table of win rates against agents

    Parameters
    ----------
    entries: Dict[str, float]
        Dictionary with agent ID's for keys and win rates for values
    float_precision: Optional[int]
        How many values to show after the decimal point
    """
    header1 = "League Agent"
    header2 = "Win Rate"

    max_string_length = max([len(header1)] + [len(key) for key in entries]) + 2

    header2_length = max([len(header2), float_precision + 7]) + 2
    divider = "+" + "-" * max_string_length + "+" + "-" * header2_length + "+"

    print(divider)

    print(
        "|"
        + centered(header1, max_string_length)
        + "|"
        + centered(header2, header2_length)
        + "|"
    )

    print(divider)

    for entry, v in entries.items():
        print(
            "|"
            + centered(entry, max_string_length)
            + "|"
            + centered(f"{v*100:0.1f}%", header2_length)
            + "|"
        )

    print(divider)


def beating_league(agent: Agent) -> bool:
    win_rates = {k: np.mean(v) for k, v in agent.win_rates.items()}
    print_table(win_rates)
    return np.mean([int(v >= 0.5) for v in win_rates.values()]) >= 0.75


async def league_match(
    challenger: OpponentPlayer,
    opponent: LeaguePlayer,
    nb_battles: Optional[int] = 100,
) -> None:
    """Asynchronous function for handling one player battling another.

    Parameters
    ----------
    challenger: OpponentPlayer
        The training agent.
    opponent: LeaguePlayer
        The player used to load in all of the League schemes.
    nb_battles: Optional[int]
        How many battles to perform.

    Returns
    -------
    None
    """
    await challenger.battle_against(opponent, n_battles=nb_battles)


def league_score(
    agent: PPOAgent,
    preprocessor: Preprocessor,
    opponent: LeaguePlayer,
    league_dir: Path,
    skill_tracker: LeagueSkillTracker,
    team_builder: AgentTeamBuilder,
    nb_battles: Optional[int] = 100,
) -> float:
    """Function for determining how many of the league agents are considered 'beaten'.

    Parameters
    ----------
    agent: PPOAgent
        The currently training agent.
    preprocessor: Preprocessor
        The preprocessing scheme for the current agent.
    opponent: LeaguePlayer
        The player for handling all of the league logic.
    league_dir: str
        The path to the league agents.
    skill_tracker: LeagueSkillTracker
        Helper class for handling the trueskill of all of the agents.
    nb_battles: Optional[int]
        The number of battles used to determine the agent's win rates. Default: 100

    Returns
    -------
    float
        Percentage of the league that the agent has over a 50% win rate against, normalized between
        0 and 1.
    """
    print("starting score")
    challenger = OpponentPlayer(
        opponent=RLOpponent(
            network=agent.network,
            preprocessor=preprocessor,
            device=agent.device,
            sample_moves=False,
        ),
        max_concurrent_battles=100,
        player_configuration=PlayerConfiguration(
            username="rlchallenger", password="rlchallenger1234"
        ),
        team=team_builder,
    )

    print("challenger made")
    sample_moves = opponent.sample_moves
    opponent.sample_moves = False

    overall_wins = 0
    win_dict = {}
    for league_agent in tqdm(league_dir.iterdir()):
        print(league_agent)
        # To eval, we play 100 games against each agent in the league. If the agent wins over 50
        # games against 75% of the opponents, it is added to the league as a league agent.
        a = opponent.change_agent(league_agent)
        print(a)
        asyncio.get_event_loop().run_until_complete(
            league_match(challenger, opponent, nb_battles=nb_battles)
        )

        for result in challenger.battle_history:
            skill_tracker.update(result, league_agent.stem)
        agent.log_scalar(
            f"League Validation/{league_agent.stem}",
            challenger.win_rate * nb_battles,
        )
        win_dict[league_agent.stem] = challenger.win_rate

        overall_wins += int(challenger.win_rate >= 0.5)
        challenger.reset_battles()
    print_table(win_dict)
    opponent.sample_moves = sample_moves
    return overall_wins / len(list(league_dir.iterdir()))


def move_to_league(
    challengers_dir: Path, league_dir: Path, tag: str, epoch: int
) -> None:
    """Adds the files necessary to build an agent into the league directory

    Parameters
    ----------
    challengers_dir: Path
        The path to the agent
    league_dir: Path
        The path to the league
    tag: str
        The name of the agent
    epoch: int
        The epoch of training
    """
    try:
        league_agent = get_save_dir(league_dir / tag, epoch, False)

        league_agent.symlink_to(
            get_save_dir(challengers_dir / tag, epoch), target_is_directory=True
        )
    except FileExistsError:
        pass


def league_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    matchmaker: MatchMaker,
    skill_tracker: LeagueSkillTracker,
    batch_size: int,
    rollout_len: int,
    epoch_len: int,
    step_counter: StepCounter,
    epoch: int,
) -> None:
    """Runs one epoch of league-style training. This function is meant to be passed into the
    player's `play_against()` function.

    Parameters
    ----------
    player: RLPlayer
        The player that is actually running the game. This acts as our "environment" and has the
        `step()` and `reset()` functions.
    agent: PPOAgent
        The agent that is playing the game. Handles the sampling of moves and also performing the
        learn step for the agent.
    opponent: LeaguePlayer
        The agent that is playing against the training agent. This agent handles scheme selection,
        but its actual action selection is being done in a separate thread.
    matchmaker: MatchMaker
        This handles the matchmaking process for the agent. It will take in the Trueskill values for
        all of the agents to determine a match-up where each agent has an even chance at winning.
    skill_tracker: skill_tracker
        This object tracks the trueskill of the agent, as well as all of the agents in the league.
    batch_size: int
        The batch size for backprop.
    rollout_len: int
        How many steps are in a rollout.
    epoch_len: int
        How many steps are in an epoch.
    step_counter: StepCounter
        Tracks the total number of steps across each epoch.
    epoch: int
        The current training epoch.

    Returns
    -------
    None
    """
    start_step = step_counter.steps
    history = History()

    while True:
        episode_start_step = step_counter.steps
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)
        episode_end_step = step_counter.steps

        agent.log_scalar(
            "Agent Outputs/Average Episode Reward",
            float(np.sum(episode.rewards)),
        )

        agent.log_scalar(
            "Agent Outputs/Average Probabilities",
            float(np.mean([np.exp(lp) for lp in episode.log_probabilities])),
        )

        agent.update_winrates(opponent.tag, int(episode.rewards[-1] > 0))

        agent.log_scalar(
            f"League Training/{opponent.tag}",
            float(np.mean(agent.win_rates[opponent.tag])),
        )

        if opponent.tag != "self":
            skill_tracker.update(episode.rewards[-1] > 0, opponent.tag)
            for k, v in skill_tracker.skill.items():
                agent.log_scalar(f"True Skill/{k}", v)

        opponent_name = matchmaker.choose_match(
            skill_tracker.agent_skill,
            skill_tracker.skill_ratings,
        )

        if opponent_name.parent.parent == matchmaker.league_path.challengers:
            opponent_name = "self"

        _ = opponent.change_agent(opponent_name)

        history.add_episode(episode)

        if episode_end_step // 100_000 != episode_start_step // 100_000:
            agent.save_model(epoch, agent.network)

        if len(history) > batch_size * rollout_len:
            history.build_dataset()
            data_loader = DataLoader(
                history,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            epoch_losses = agent.learn_step(data_loader)
            opponent.update_network(agent.network)

            for k, v in epoch_losses.items():
                agent.log_scalar(f"League Loss/{k}", np.mean(v))

            history.free_memory()

            if step_counter.steps - start_step >= epoch_len:
                break


def league_play(
    preprocessor: Preprocessor,
    network: nn.Module,
    league_path: PokePath,
    args: Dict[str, Any],
    starting_epoch: Optional[int] = 0,
):
    """Main loop for training a league agent.

    Parameters
    ----------
    league_path
    preprocessor
        The preprocessor that this agent will be using to convert Battle objects to tensors.
    network
        The network that will be training.
    args
        Hyperparameters used for training. MUST CONTAIN:
        - batch_size: int
        - battle_format: str
        - clip: float
        - device: int
        - entropy_weight: float
        - epoch_len: int
        - league_play_prob: float
        - logdir: str
        - lr: float
        - nb_steps: int
        - rewards: Dict[str, float]
        - sample_moves: float
        - self_play_prob: float
        - tag: str
    starting_epoch
        If we're resuming, this is the epoch we're resuming from.

    Returns
    -------
    None
    """

    agent = PPOAgent(
        device=args["device"],
        network=network,
        lr=args["lr"],
        entropy_weight=args["entropy_weight"],
        clip=args["clip"],
        league_path=league_path,
        tag=args["tag"],
    )

    step_counter = StepCounter()
    skill_tracker = LeagueSkillTracker(league_path, args["resume"])
    matchmaker = MatchMaker(
        args["self_play_prob"], args["league_play_prob"], league_path
    )
    team_builder = AgentTeamBuilder(
        agent_path=league_path.agent, battle_format=args["battle_format"]
    )

    team_builder.save_team()
    player = RLPlayer(
        battle_format=args["battle_format"],
        preprocessor=preprocessor,
        reward_scheme=RewardScheme(args["rewards"]),
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
        sample_moves=args["sample_moves"],
        max_concurrent_battles=10,
        server_configuration=DockerServerConfiguration,
        team=AgentTeamBuilder(),
        training_team=team_builder,
        battle_format=args["battle_format"],
        player_configuration=PlayerConfiguration(
            username=f"rlopponent", password="rlopponent1234"
        ),
    )

    for epoch in range(starting_epoch, args["nb_steps"] // args["epoch_len"]):
        save_args(agent_dir=league_path.agent, args=args, epoch=epoch)

        player.play_against(
            env_algorithm=league_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "matchmaker": matchmaker,
                "skill_tracker": skill_tracker,
                "batch_size": args["batch_size"],
                "rollout_len": args["rollout_len"],
                "epoch_len": args["epoch_len"],
                "step_counter": step_counter,
                "epoch": epoch,
            },
        )

        agent.save_model(epoch, network)
        skill_tracker.save_skill_ratings(epoch)

        if beating_league(agent):
            move_to_league(
                challengers_dir=league_path.challengers,
                league_dir=league_path.league,
                tag=agent.tag,
                epoch=epoch,
            )

        # league_win_rate = league_score(
        #     agent,
        #     preprocessor,
        #     opponent,
        #     league_path.league,
        #     skill_tracker,
        #     team_builder,
        # )
        #
        # if league_win_rate > 0.75:
        #     move_to_league(
        #         challengers_dir=league_path.challengers,
        #         league_dir=league_path.league,
        #         tag=agent.tag,
        #         epoch=epoch,
        #     )


def main(args: Dict[str, Any]):
    league_path = PokePath(args["logdir"], args["tag"])

    network, preprocessor = build_network_and_preproc(args)

    if "resume" in args and args["resume"]:
        network.resume(league_path.agent)

    league_play(
        preprocessor=preprocessor, network=network, league_path=league_path, args=args
    )


if __name__ == "__main__":
    main(parse_args(__file__))
