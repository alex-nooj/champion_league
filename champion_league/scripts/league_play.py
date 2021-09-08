import asyncio
import os
import time
from typing import Dict
from typing import Optional

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.ppo import PPOAgent
from champion_league.env import OpponentPlayer
from champion_league.env.league_player import LeaguePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.matchmaking.skill_tracker import SkillTracker
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.preprocessors import Preprocessor
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.directory_utils import get_save_dir
from champion_league.utils.parse_args import parse_args
from champion_league.utils.progress_bar import centered
from champion_league.utils.replay import Episode
from champion_league.utils.replay import History


class StepCounter:
    def __init__(self, reporting_freq: Optional[int] = 10_000):
        """Helper class for counting the number of steps. Prints out the step number and the steps
        per second at each interval, decided by `reporting_freq`.

        Parameters
        ----------
        reporting_freq: Optional[int]
            How often to report the number of steps and the steps per second. Default: 10,000
        """
        self.steps = 0
        self.starting_time = time.time()
        self.reporting_freq = reporting_freq

    def __call__(self):
        """Call method for the StepCounter. Increases the step count and reports it if we've hit
        the reporting frequency."""
        self.steps += 1
        if self.steps % self.reporting_freq == 0:
            steps_per_sec = self.reporting_freq / (time.time() - self.starting_time)
            print(f"\nStep {self.steps}: {steps_per_sec} steps/sec")
            self.starting_time = time.time()


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


async def league_match(
    challenger: OpponentPlayer, opponent: LeaguePlayer, nb_battles: Optional[int] = 100
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
    league_dir: str,
    skill_tracker: SkillTracker,
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
    skill_tracker: SkillTracker
        Helper class for handling the trueskill of all of the agents.
    nb_battles: Optional[int]
        The number of battles used to determine the agent's win rates. Default: 100

    Returns
    -------
    float
        Percentage of the league that the agent has over a 50% win rate against, normalized between
        0 and 1.
    """
    challenger = OpponentPlayer(
        opponent=RLOpponent(
            network=agent.network,
            preprocessor=preprocessor,
            device=f"cuda:{agent.device}",
            sample_moves=False,
        ),
        max_concurrent_battles=100,
    )

    sample_moves = opponent.sample_moves
    opponent.sample_moves = False

    league_agents = os.listdir(league_dir)

    overall_wins = 0
    win_dict = {}
    for league_agent in tqdm(league_agents):
        # To eval, we play 100 games against each agent in the league. If the agent wins over 50
        # games against 75% of the opponents, it is added to the league as a league agent.
        _ = opponent.change_agent(os.path.join(league_dir, league_agent))

        asyncio.get_event_loop().run_until_complete(
            league_match(challenger, opponent, nb_battles=nb_battles)
        )

        for result in challenger.battle_history:
            skill_tracker.update(result, league_agent)
        agent.write_to_tboard(
            f"League Validation/{league_agent}", challenger.win_rate * nb_battles
        )
        win_dict[league_agent] = challenger.win_rate

        overall_wins += int(challenger.win_rate >= 0.5)
        challenger.reset_battles()
    print_table(win_dict)
    opponent.sample_moves = sample_moves
    return overall_wins / len(league_agents)


def move_to_league(challengers_dir: str, league_dir: str, tag: str, epoch: int) -> None:
    """Adds the files necessary to build an agent into the league directory

    Parameters
    ----------
    challengers_dir: str
        The path to the agent
    league_dir: str
        The path to the league
    tag: str
        The name of the agent
    epoch: int
        The epoch of training
    """
    try:
        os.symlink(
            src=get_save_dir(challengers_dir, tag, epoch),
            dst=os.path.join(
                league_dir,
                f"{tag}_{epoch:05d}",
            ),
            target_is_directory=True,
        )
    except FileExistsError:
        pass


def league_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    matchmaker: MatchMaker,
    skill_tracker: SkillTracker,
    batch_size: int,
    rollout_len: int,
    epoch_len: int,
    step_counter: StepCounter,
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

    Returns
    -------
    None
    """
    start_step = step_counter.steps
    history = History()
    observation = player.reset()
    episode = Episode()
    while True:
        action, log_prob, value = agent.sample_action(observation)

        new_observation, reward, done, info = player.step(action)
        step_counter()

        episode.append(
            observation=observation,
            action=action,
            reward=reward,
            value=value,
            log_probability=log_prob,
            reward_scale=6,
        )

        observation = new_observation

        if done:
            episode.end_episode(last_value=0)
            agent.write_to_tboard(
                "Agent Outputs/Average Episode Reward", float(np.sum(episode.rewards))
            )

            agent.write_to_tboard(
                "Agent Outputs/Average Probabilities",
                float(np.mean([np.exp(lp) for lp in episode.log_probabilities])),
            )

            agent.update_winrates(opponent.tag, int(reward > 0))

            agent.write_to_tboard(
                f"League Training/{opponent.tag}",
                float(
                    agent.win_rates[opponent.tag][0] / agent.win_rates[opponent.tag][1]
                ),
            )

            if opponent.tag != "self":
                skill_tracker.update(reward > 0, opponent.tag)
                for k, v in skill_tracker.skill.items():
                    agent.write_to_tboard(f"True Skill/{k}", v)

            history.add_episode(episode)

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

                for key in epoch_losses:
                    for val in epoch_losses[key]:
                        agent.write_to_tboard(f"League Loss/{key}", val)

                history.free_memory()

            if step_counter.steps - start_step >= epoch_len:
                break
            else:
                opponent_name = matchmaker.choose_match(
                    skill_tracker.agent_skill,
                    skill_tracker.skill_ratings,
                )
                _ = opponent.change_agent(opponent_name)
                observation = player.reset()
                episode = Episode()


def league_play(
    battle_format: str,
    preprocessor: Preprocessor,
    sample_moves: bool,
    agent: PPOAgent,
    matchmaker: MatchMaker,
    skill_tracker: SkillTracker,
    nb_steps: int,
    epoch_len: int,
    batch_size: int,
    args: DotDict,
    logdir: str,
    rollout_len: int,
    starting_epoch: Optional[int] = 0,
):
    """The main loop for running the league.

    Parameters
    ----------
    battle_format: str
    preprocessor: Preprocessor
    sample_moves: bool
    agent: PPOAgent
        The agent that is playing the game. Handles the sampling of moves and also performing the
        learn step for the agent.
    matchmaker: MatchMaker
        This handles the matchmaking process for the agent. It will take in the Trueskill values for
        all of the agents to determine a match-up where each agent has an even chance at winning.
    skill_tracker: skill_tracker
        This object tracks the trueskill of the agent, as well as all of the agents in the league.
    nb_steps: int
        The total number of steps to train the agent.
    epoch_len: int
        How many steps are in an epoch.
    batch_size: int
        The batch size for backprop.
    args: DotDict
    logdir: str
    rollout_len: int
    starting_epoch: Optional[int]

    Returns
    -------

    """
    agent.save_args(args)
    step_counter = StepCounter()

    for epoch in range(starting_epoch, nb_steps // epoch_len):
        agent.save_model(agent.network, epoch, args)
        skill_tracker.save_skill_ratings(epoch)

        player = RLPlayer(
            battle_format=battle_format,
            embed_battle=preprocessor.embed_battle,
        )

        opponent = LeaguePlayer(
            device=agent.device,
            network=agent.network,
            preprocessor=preprocessor,
            sample_moves=sample_moves,
            max_concurrent_battles=10,
        )

        player.play_against(
            env_algorithm=league_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "matchmaker": matchmaker,
                "skill_tracker": skill_tracker,
                "batch_size": batch_size,
                "rollout_len": rollout_len,
                "epoch_len": epoch_len,
                "step_counter": step_counter,
            },
        )

        league_win_rate = league_score(
            agent, preprocessor, opponent, os.path.join(logdir, "league"), skill_tracker
        )

        if league_win_rate > 0.75:
            move_to_league(
                challengers_dir=os.path.join(logdir, "challengers"),
                league_dir=os.path.join(logdir, "league"),
                tag=agent.tag,
                epoch=epoch,
            )

        del player
        del opponent

    agent.save_model(agent.network, nb_steps // epoch_len, args)
    skill_tracker.save_skill_ratings(nb_steps // epoch_len)


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    matchmaker = MatchMaker(
        args.self_play_prob, args.league_play_prob, args.logdir, args.tag
    )

    agent = PPOAgent(
        device=args.device,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=os.path.join(args.logdir, "challengers"),
        tag=args.tag,
    )

    agent.save_args(args)

    skill_tracker = SkillTracker.from_args(args)

    league_play(
        args.battle_format,
        preprocessor,
        args.sample_moves,
        agent,
        matchmaker,
        skill_tracker,
        args.nb_steps,
        args.epoch_len,
        args.batch_size,
        args,
        args.logdir,
        args.rollout_len,
    )


if __name__ == "__main__":
    main(parse_args())
