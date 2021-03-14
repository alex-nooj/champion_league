"""
Train a PPO-based agent in the league

Usage:
    train_ppo [options]

Options
    --agent <str>                   Python path to agent class [default: champion_league.agent.dqn.DQNAgent]
    --tag <str>                     Name of the agent [default: None]
    --opponent_device <int>         GPU to load the opponent onto [default: None]
    --p_exploit <float>             % of time to play exploiters [default: 0]
    --p_league <float>              % of time to play league agents [default: 0.2]
    --logdir <str>                  Path to agents [default: /home/alex/Documents/pokemon_trainers/]
    --nb_train_steps <int>          Number of steps to train for [default: 10_000_000]
    --epoch_len <int>               Number of games to play before saving an agent [default: 1_000_000]
    --batch_size <int>              Batch size [default: 64]
    --rollout_len <int>             Length of the rollouts [default: 128]
    --device <int>                  GPU to load the training agent on [default: 0]
    --agent_type <str>              What type of agent we're training [default: challengers]
    --learning_rate <float>         Learning rate of the agent [default: 0.001]
    --discount <float>              Rewards discount [default: 0.99]
    --entropy_weight <float>        Value to weight entropy loss by [default: 0.01]
    --gae_discount <float>          Value to weight the generalize advantage entropy by [default: 0.95]
    --policy_clipping <float>       Clip values for PPO [default: 0.2]
    --nb_rollout_epoch <int>        How many epochs to do on the minibatches [default: 4]
    --rollout_minibatch_len <int>   How long the minibatch rollouts should be [default: 32]
"""
import os
import time
from datetime import datetime

import torch
from adept.utils.util import DotDict
from poke_env.player_configuration import PlayerConfiguration
from torch import nn
import torch.nn.functional as F

from champion_league.agent.base.base_agent import Agent
from champion_league.agent.ppo.agent import PPOAgent
from champion_league.agent.league.agent import LeaguePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.network.lstm_network import LSTMNetwork


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
    args.nb_train_steps = int(args.nb_train_steps)
    args.epoch_len = int(args.epoch_len)
    args.batch_size = int(args.batch_size)
    if args.device != "None":
        args.device = f"cuda:{int(args.device)}"
    else:
        args.device = "cpu"

    args.learning_rate = float(args.learning_rate)
    args.discount = float(args.discount)
    args.entropy_weight = float(args.entropy_weight)
    args.gae_discount = float(args.gae_discount)
    args.policy_clipping = float(args.policy_clipping)
    args.nb_rollout_epoch = int(args.nb_rollout_epoch)
    args.rollout_minibatch_len = int(args.rollout_minibatch_len)
    args.rollout_len = int(args.rollout_len)
    args.nb_steps = 0
    args.epoch = 0
    return args


def league_is_beaten(win_rates):
    win_percentages = [win_rates[key][0] / win_rates[key][1] for key in win_rates]
    return all(x >= 0.7 for x in win_percentages)


def add_to_league(args, epoch):
    os.symlink(
        os.path.join(args.logdir, "challengers", args.tag, f"{args.tag}_{epoch}"),
        os.path.join(args.logdir, "league", f"{args.tag}_{epoch}")
    )


def run(
        player: RLPlayer,
        agent: PPOAgent,
        opponent: LeaguePlayer,
        network: nn.Module,
        args: DotDict
):
    """
    Runs the training loop for an agent

    Parameters
    ----------
    player: RLPlayer
        The player that is handling the environment
    agent: PPOAgent
        Agent that handles the learning and acting
    opponent: LeaguePlayer
        Player that handles all of the league data
    network: nn.Module
        Neural network doing the predicting/learning
    args: DotDict
        Various arguments

    Returns
    -------

    """
    epoch = args.epoch
    total_nb_steps = args.nb_steps
    total_win_rates = {}

    done = False
    state = player.reset()
    agent.reset()
    rolling_reward = 0
    rollout_steps = 0
    next_states = []
    learn_internals = {"hx": [], "cx": []}
    internals = {
        k: r.view(1, -1).to(args.device)
        for k, r in network.new_internals().items()
    }
    optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
    start_time = time.time()
    while total_nb_steps < args.nb_train_steps:
        state = state.float().to(args.device)

        # TODO: Need the agent's log probabilities
        pred, next_internals = network(state, internals)

        action = agent.choose_move(pred["action"])

        next_state, reward, done, info = player.step(action.item())

        if rollout_steps == args.rollout_len:
            next_states.append(next_state)
            learn_internals["hx"].append(next_internals["hx"])
            learn_internals["cx"].append(next_internals["cx"])
            rollout_steps = 0
        else:
            logit = pred["action"].view(pred["action"].shape[0], -1)
            log_softmax = F.log_softmax(logit, dim=1)
            log_prob = log_softmax.gather(1, action)

            agent.memory.push(
                state=state,
                next_state=next_state,
                action=action,
                value=pred["critic"],
                reward=reward,
                terminal=done,
                log_probs=log_prob,
                hx=internals["hx"],
                cx=internals["cx"],
                step_nb=total_nb_steps
            )
            rollout_steps += 1

        state = next_state
        internals = next_internals

        if len(next_states) == args.batch_size and total_nb_steps > 0:
            loss = agent.learn_step(
                optimizer,
                network,
                torch.stack(next_states).to(args.device),
                {
                    "hx": torch.stack(learn_internals["hx"]).squeeze(1).to(args.device),
                    "cx": torch.stack(learn_internals["cx"]).squeeze(1).to(args.device)
                }
            )
            agent.log_to_tensorboard(total_nb_steps, loss=loss)
            agent.reset()

        total_nb_steps += 1
        rolling_reward += reward

        if total_nb_steps % args.epoch_len == 0:
            epoch += 1
            agent.save_model(network, epoch)
            args.nb_steps = total_nb_steps
            agent.save_args(epoch)

            agent.save_wins(epoch, agent.win_rates)
            if league_is_beaten(agent.win_rates):
                add_to_league(args, epoch)
                print("League beaten!")
            agent.win_rates = {}

        if total_nb_steps % 100 == 0:
            steps_per_sec = 100 / (time.time() - start_time)
            print(f"STEP: {total_nb_steps}" +
                  f"({opponent.current_agent.tag}) {steps_per_sec: 0.3f} steps/sec " +
                  f"REWARD: {rolling_reward / 100: 0.1f}")
            start_time = time.time()
            rolling_reward = 0

        if done:
            if opponent.current_agent.tag not in agent.win_rates:
                agent.win_rates[opponent.current_agent.tag] = [info["won"], 1]
            else:
                agent.win_rates[opponent.current_agent.tag][0] += info["won"]
                agent.win_rates[opponent.current_agent.tag][1] += 1

            agent.log_to_tensorboard(
                total_nb_steps,
                win_rates={opponent.current_agent.tag: agent.win_rates[opponent.current_agent.tag]},
                reward=reward
            )

            opponent.change_agent(agent.win_rates)
            state = player.reset()


def main(args: DotDict):
    args.nb_actions = len(list(range(4 * 4 + 6)))

    if not os.path.isdir(os.path.join(args.logdir, "exploiters", args.tag)):
        os.mkdir(os.path.join(args.logdir, "exploiters", args.tag))

    agent = PPOAgent(
        args=args,
        rollout_len=args.rollout_len,
        batch_size=args.batch_size,
        discount=args.discount,
        entropy_weight=args.entropy_weight,
        gae_discount=args.gae_discount,
        policy_clipping=args.policy_clipping,
        device=args.device,
        nb_rollout_epoch=args.nb_rollout_epoch,
        rollout_minibatch_len=args.rollout_minibatch_len
    )

    network = LSTMNetwork(args.nb_actions)
    network = network.to(args.device)
    network.eval()
    agent.save_model(network, 0)
    agent.save_args(0)
    env_player = RLPlayer(
        embed_battle=network.embed_battle,
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration(args.tag, None),
    )

    opponent = LeaguePlayer(args)

    env_player.play_against(
        env_algorithm=run,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "network": network,
            "args": args,
            "opponent": opponent
        }
    )


if __name__ == "__main__":
    main(parse_args())
