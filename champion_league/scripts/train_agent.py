"""
Train an agent in the league

Usage:
    train_agent [options]

Options
    --agent <str>               Python path to agent class [default: champion_league.agent.dqn.DQNAgent]
    --tag <str>                 Name of the agent [default: None]
    --opponent_device <int>     GPU to load the opponent onto [default: None]
    --p_exploit <float>         % of time to play exploiters [default: 0]
    --p_league <float>          % of time to play league agents [default: 0.2]
    --logdir <str>              Path to agents [default: /home/alex/Documents/pokemon_trainers/]
    --network <str>             Python path to network [default:
    --nb_train_episodes <int>   Number of games to train for [default: 10_000_000]
    --epoch_len <int>           Number of games to play before saving an agent [default: 1_000_000]
    --batch_size <int>          Batch size [default: 256]
    --device <int>              GPU to load the training agent on [default: 0]
    --agent_type <str>          What type of agent we're training [default: challengers]
"""
import os
import time
from datetime import datetime

import torch
from adept.utils.util import DotDict
from poke_env.player_configuration import PlayerConfiguration

from champion_league.agent.dqn import DQNAgent
from champion_league.agent.league.league_player import LeaguePlayer
from champion_league.env.rl_player import RLPlayer


def parse_args() -> DotDict:
    from docopt import docopt

    args = docopt(__doc__)
    args = {k.strip("--").replace("-", "_"): v for k, v in args.items()}

    args = DotDict(args)
    args.agent = str(args.agent)

    if args.tag != "None":
        args.tag = str(args.tag)
    else:
        args.tag = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    if args.opponent_device != "None":
        args.opponent_device = f"cuda:{int(args.opponent_device)}"
    else:
        args.opponent_device = "cpu"

    args.p_exploit = float(args.p_exploit)
    args.p_league = float(args.p_league)

    args.logdir = str(args.logdir)
    args.network = str(args.network)
    args.nb_train_episodes = int(args.nb_train_episodes)
    args.epoch_len = int(args.epoch_len)
    args.batch_size = int(args.batch_size)
    if args.device != "None":
        args.device = f"cuda:{int(args.device)}"
    else:
        args.device = "cpu"

    args.nb_steps = 0
    return args


def league_is_beaten(win_rates):
    win_percentages = [win_rates[key][0] / win_rates[key][1] for key in win_rates]
    return all(x >= 0.7 for x in win_percentages)


def add_to_league(args, epoch):
    os.symlink(
        os.path.join(args.logdir, "challengers", args.tag, f"{args.tag}_{epoch}"),
        os.path.join(args.logdir, "league", f"{args.tag}_{epoch}"),
    )


def run(player, agent, opponent, args):
    epoch = 0
    total_win_rates = {}
    agent.save_model(agent.network, epoch)
    agent.save_args(epoch)
    total_nb_steps = 0
    profile = False
    for i_episode in range(args.nb_train_episodes):
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
                args.nb_steps = total_nb_steps
                agent.save_args(epoch)

                agent.save_wins(epoch, agent.win_rates)
                if league_is_beaten(agent.win_rates):
                    add_to_league(args, epoch)
                    print("League beaten!")
                agent.win_rates = {}
            # state = player.reset()
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
            win_rates={opponent.current_agent.tag: total_win_rates[opponent.current_agent.tag]},
            reward=reward,
        )

        opponent.change_agent(agent.win_rates)

    player.complete_current_battle()


def main(args: DotDict):
    # agent_cls = importlib.import_module(args.agent)
    args.nb_actions = len(list(range(4 * 4 + 6)))

    if not os.path.isdir(os.path.join(args.logdir, "exploiters", args.tag)):
        os.mkdir(os.path.join(args.logdir, "exploiters", args.tag))

    agent = DQNAgent(args)
    agent.save_model(agent.network, 0)

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
