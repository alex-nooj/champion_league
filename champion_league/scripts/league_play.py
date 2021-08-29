import json
import os
import time
from typing import Dict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from champion_league.agent.opponent.league_player import LeaguePlayer
from champion_league.agent.ppo import PPOAgent
from champion_league.env.rl_player import RLPlayer
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.matchmaking.skill_tracker import SkillTracker
from champion_league.network import build_network_from_args
from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.directory_utils import DotDict, get_save_dir
from champion_league.utils.parse_args import parse_args
from champion_league.utils.progress_bar import centered
from champion_league.utils.replay import Episode
from champion_league.utils.replay import History


class StepCounter:
    def __init__(self, reporting_freq: Optional[int] = 10_000):
        self.steps = 0
        self.starting_time = time.time()
        self.reporting_freq = reporting_freq

    def __call__(self):
        self.steps += 1
        if self.steps % self.reporting_freq == 0:
            steps_per_sec = self.reporting_freq / (
                time.time() - self.starting_time
            )
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


def league_check(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    logdir: str,
    epoch: int,
    args: DotDict,
    nb_games: Optional[int] = 100
) -> None:
    """Function for determining if an agent has met the minimum requirements needed to be admitted
    into the league

    Parameters
    ----------
    player: RLPlayer
        The player, used for communicating with the server
    agent: PPOAgent
        The agent that handles acting and contains the learn step
    opponent: LeaguePlayer
        The opponent that the agent is playing against
    logdir: str
        Where the agent is stored
    epoch: int
        Which epoch this is
    args: DotDict
        The arguments used to construct the agent
    nb_games: Optional[int]
        How many games to play against each agent. Default: 100
    """

    # We want this to be an evaluation battle, so we temporarily ensure that the opponent is not
    # sampling its moves.
    sample_moves = opponent.sample_moves
    opponent.sample_moves = False

    # Grab all of the league agents
    league_agents = os.listdir(os.path.join(logdir, "league"))

    overall_wins = 0
    win_dict = {}
    for league_agent in league_agents:
        # To eval, we play 100 games against each agent in the league. If the agent wins over 50
        # games against 75% of the opponents, it is added to the league as a league agent.
        _ = opponent.change_agent(os.path.join(logdir, "league", league_agent))

        # The battle loop
        win_rate = 0
        for episode in range(nb_games):
            done = False
            reward = 0
            observation = player.reset()
            while not done:
                observation = observation.float().to(agent.device)

                action, log_prob, value = agent.sample_action(observation)

                observation, reward, done, info = player.step(action)
            # Check if the agent won
            win_rate += int(reward > 0)

        agent.write_to_tboard(f"League Validation/{league_agent}", win_rate)
        win_dict[league_agent] = win_rate / nb_games

        # Check if the agent has won at least 50% of the games against this agent.
        overall_wins += int(win_rate >= (nb_games / 2))

    # If the agent has a win rate over 50% for 75% of the league agents, it is then added to the
    # league.
    if overall_wins / len(league_agents) >= 0.75:
        move_to_league(logdir, agent.tag, epoch, args, agent.network)

    # Print the table with all the win rates, complete the battle, and reset the opponent
    print_table(win_dict)
    player.complete_current_battle()
    opponent.sample_moves = sample_moves


def move_to_league(
    logdir: str, tag: str, epoch: int, args: DotDict, network: torch.nn.Module
) -> None:
    """Adds the files necessary to build an agent into the league directory

    Parameters
    ----------
    logdir: str
        The path to the agent
    tag: str
        The name of the agent
    epoch: int
        The epoch of training
    args: DotDict
        The arguments used to construct the agent
    network: torch.nn.Module
        The neural network being trained
    """
    try:
        os.symlink(
            src=get_save_dir(os.path.join(logdir, "challengers"), tag, epoch),
            dst=os.path.join(
                logdir,
                "league",
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
    skilltracker: SkillTracker,
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
    skilltracker: SkillTracker
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
        observation = observation.float().to(agent.device)

        action, log_prob, value = agent.sample_action(observation)

        new_observation, reward, done, info = player.step(action)
        step_counter()

        episode.append(
            observation=observation.squeeze(),
            action=action,
            reward=reward,
            value=value,
            log_probability=log_prob,
            reward_scale=6.0,
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
                skilltracker.update(reward > 0, opponent.tag)
                for k, v in skilltracker.skill.items():
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
                    skilltracker.agent_skill,
                    skilltracker.skill_ratings,
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
    skilltracker: SkillTracker,
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
    skilltracker: SkillTracker
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
        skilltracker.save_skill_ratings(epoch)

        player = RLPlayer(
            battle_format=battle_format,
            embed_battle=preprocessor.embed_battle,
        )

        opponent = LeaguePlayer(
            device=agent.device,
            network=agent.network,
            preprocessor=preprocessor,
            sample_moves=sample_moves,
        )

        # Check to see if the agent is able to enter the league or not
        player.play_against(
            env_algorithm=league_check,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "logdir": logdir,
                "epoch": epoch,
                "args": args,
            },
        )

        player.reset_battles()
        opponent.reset_battles()

        player.play_against(
            env_algorithm=league_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "matchmaker": matchmaker,
                "skilltracker": skilltracker,
                "batch_size": batch_size,
                "rollout_len": rollout_len,
                "epoch_len": epoch_len,
                "step_counter": step_counter,
            },
        )

        del player
        del opponent
    agent.save_model(agent.network, nb_steps // epoch_len, args)
    skilltracker.save_skill_ratings(nb_steps // epoch_len)

    player = RLPlayer(
        battle_format=battle_format,
        embed_battle=preprocessor.embed_battle,
    )

    opponent = LeaguePlayer(
        device=agent.device,
        network=agent.network,
        preprocessor=preprocessor,
        sample_moves=sample_moves,
    )

    # Check to see if the agent is able to enter the league or not
    player.play_against(
        env_algorithm=league_check,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "opponent": opponent,
            "logdir": logdir,
            "epoch": epoch,
            "args": args,
        },
    )


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

    skilltracker = SkillTracker.from_args(args)

    league_play(
        args.battle_format,
        preprocessor,
        args.sample_moves,
        agent,
        matchmaker,
        skilltracker,
        args.nb_steps,
        args.epoch_len,
        args.batch_size,
        args,
        args.logdir,
        args.rollout_len,
    )


if __name__ == "__main__":
    main(parse_args())
