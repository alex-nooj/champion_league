"""
Resume training on a stopped agent

Usage:
    resume_training [options]

Options
    --tag <str>                 Name of the agent [default: None]
    --opponent_device <int>     GPU to load the opponent onto [default: None]
    --logdir <str>              Path to agents [default: /home/anewgent/Documents/pokemon_trainers]
    --nb_train_episodes <int>   Number of games to train for [default: 10_000_000]
    --epoch_len <int>           Number of steps to take before saving an agent [default: 1_000_000]
    --batch_size <int>          Batch size [default: 64]
    --device <int>              GPU to load the training agent on [default: 0]
    --agent_type <str>          What type of agent we're training [default: challengers]
"""
import json
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from adept.utils.util import DotDict
from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.dqn import DQNAgent
from champion_league.agent.league.agent import LeaguePlayer
from champion_league.env.rl_player import RLPlayer


def parse_args() -> DotDict:
    from docopt import docopt

    args = docopt(__doc__)
    args = {k.strip("--").replace("-", "_"): v for k, v in args.items()}

    args = DotDict(args)

    if args.tag != "None":
        args.tag = str(args.tag)
    else:
        raise RuntimeError("Tag cannot be none! Must specify a trainer!")

    if args.opponent_device != "None":
        args.opponent_device = f"cuda:{int(args.opponent_device)}"
    else:
        args.opponent_device = "cpu"

    args.logdir = str(args.logdir)
    args.nb_train_episodes = int(args.nb_train_episodes)
    args.epoch_len = int(args.epoch_len)
    args.batch_size = int(args.batch_size)
    if args.device != "None":
        args.device = f"cuda:{int(args.device)}"
    else:
        args.device = "cpu"

    # Get the old args
    agent_epochs = [
        int(agent.rsplit("_")[-1])
        for agent in os.listdir(os.path.join(args.logdir, args.agent_type, args.tag))
        if args.tag in agent
    ]
    epoch_num = sorted(agent_epochs)[-1]

    with open(
        os.path.join(
            args.logdir,
            args.agent_type,
            args.tag,
            f"{args.tag}_{epoch_num}",
            "args.json",
        ),
        "r",
    ) as fp:
        old_args = json.load(fp)

    for key in old_args:
        if key in args:
            old_args[key] = args[key]
    old_args["epoch_num"] = epoch_num
    print(f"Resuming {args.tag}_{epoch_num}")
    return DotDict(old_args)


def league_is_beaten(win_rates):
    win_percentages = [win_rates[key][0] / win_rates[key][1] for key in win_rates]
    return all(x >= 0.7 for x in win_percentages)


def add_to_league(args, epoch):
    os.symlink(
        os.path.join(args.logdir, "challengers", args.tag, f"{args.tag}_{epoch}"),
        os.path.join(args.logdir, "league", f"{args.tag}_{epoch}"),
    )


def run(player, agent, opponent, args):
    epoch = args.epoch_num
    total_win_rates = {}
    agent.save_model(agent.network, epoch)
    agent.save_args(epoch)
    total_nb_steps = args.nb_steps
    profile = False
    for i_episode in range(args.epoch_num * 1000, args.nb_train_episodes):
        state = player.reset()
        done = False
        info = {}
        nb_steps = 0
        start_time = time.time()
        while not done:
            state = state.float().to(args.device)
            action = agent.choose_move(state.view(1, -1))

            next_state, reward, done, info = player.step(action.item())
            reward = torch.tensor([reward], device=args.device)

            agent.memory.push(state, action, next_state.double(), reward)
            state = next_state

            loss = agent.learn_step(profile)
            nb_steps += 1
            total_nb_steps += 1

            agent.log_to_tensorboard(total_nb_steps, loss=loss)

            if total_nb_steps % args.epoch_len == 0:
                epoch += 1
                agent.save_model(agent.network, epoch)

                agent.save_args(epoch)

                agent.save_wins(epoch, agent.win_rates)
                if league_is_beaten(agent.win_rates):
                    add_to_league(args, epoch)
                    print("League beaten!")
                agent.win_rates = {}

        end_time = time.time()
        steps_per_sec = nb_steps / (end_time - start_time)
        print(
            f"{i_episode} ({opponent.current_agent.tag}): {steps_per_sec: 0.3f} steps/sec, REWARD: {int(reward[0])}"
        )
        profile = steps_per_sec < 20.0
        if opponent.current_agent.tag not in agent.win_rates:
            agent.win_rates[opponent.current_agent.tag] = [info["won"], 1]
        else:
            agent.win_rates[opponent.current_agent.tag][0] += info["won"]
            agent.win_rates[opponent.current_agent.tag][1] += 1

        if opponent.current_agent.tag not in total_win_rates:
            total_win_rates[opponent.current_agent.tag] = [info["won"], 1]
        else:
            total_win_rates[opponent.current_agent.tag][0] += info["won"]
            total_win_rates[opponent.current_agent.tag][1] += 1

        agent.log_to_tensorboard(
            total_nb_steps,
            win_rates={
                opponent.current_agent.tag: total_win_rates[opponent.current_agent.tag]
            },
            reward=reward,
        )
        nb_wins = np.sum([total_win_rates[key][0] for key in total_win_rates])
        nb_games = np.sum([total_win_rates[key][1] for key in total_win_rates])
        agent.log_to_tensorboard(
            total_nb_steps,
            win_rates={"total": [nb_wins, nb_games]},
        )

        opponent.change_agent(agent.win_rates)

    player.complete_current_battle()


def main(args: DotDict):
    # agent_cls = importlib.import_module(args.agent)
    args.nb_actions = len(list(range(4 * 4 + 6)))

    if not os.path.isdir(os.path.join(args.logdir, "exploiters", args.tag)):
        os.mkdir(os.path.join(args.logdir, "exploiters", args.tag))

    agent = DQNAgent(args)
    agent.network = agent.load_model(
        agent.network,
        os.path.join(
            args.logdir,
            args.agent_type,
            args.tag,
            f"{args.tag}_{args.epoch_num}",
            f"{args.tag}_{args.epoch_num}.pt",
        ),
    )
    env_player = RLPlayer(
        embed_battle=agent.network.embed_battle,
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration(args.tag, None),
    )

    opponent = LeaguePlayer(args)

    env_player.play_against(
        env_algorithm=run,
        opponent=opponent,
        env_algorithm_kwargs={"agent": agent, "args": args, "opponent": opponent},
    )


if __name__ == "__main__":
    main(parse_args())
