import json
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from adept.utils.util import DotDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from champion_league.agent.opponent.league_player import LeaguePlayer
from champion_league.agent.ppo import PPOAgent
from champion_league.env.rl_player import RLPlayer
from champion_league.matchmaking.matchmaker import MatchMaker
from champion_league.network import build_network_from_args
from champion_league.ppo.replay import History, Episode
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.parse_args import parse_args
from champion_league.utils.progress_bar import centered


def print_table(entries: Dict[str, float], float_precision: Optional[int] = 1):
    header1 = "League Agent"
    header2 = "Win Rate"

    max_string_length = max(
        [len(header1)] + [len(key) for key in entries]
    ) + 2

    header2_length = float_precision + 7
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
) -> None:
    # Grab all of the league agents
    league_agents = os.listdir(os.path.join(logdir, "league"))

    overall_wins = 0
    win_dict = {}
    for league_agent in league_agents:
        _ = opponent.change_agent(os.path.join(logdir, "league", league_agent))

        win_rate = 0
        for episode in range(100):
            done = False
            reward = 0
            observation = player.reset()
            while not done:
                observation = observation.float().to(agent.device)

                action, log_prob, value = agent.sample_action(observation)

                observation, reward, done, info = player.step(action)
            win_rate += int(reward > 0)
        agent.write_to_tboard(f"League Validation/{league_agent}", win_rate)
        win_dict[league_agent] = win_rate / 100
        overall_wins += int(win_rate >= 50)

    if overall_wins / len(league_agents) >= 0.75:
        move_to_league(logdir, agent.tag, epoch, args, agent.network)
    print_table(win_dict)


def move_to_league(logdir: str, tag: str, epoch: int, args: DotDict, network: torch.nn.Module):
    save_dir = os.path.join(logdir, "league", f"{tag}_{epoch:07d}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, "args.json"), "w") as fp:
        json.dump(args, fp, indent=2)

    torch.save(network.state_dict(), os.path.join(save_dir, "network.pt"))


def league_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    matchmaker: MatchMaker,
    epoch_len: int,
    batch_size: int,
    rollout_len: int,
):
    history = History()

    epoch_loss_count = {}

    done = True
    observation = None
    episode = Episode()
    win_rates = {}

    for step in tqdm(range(epoch_len)):
        if done:
            observation = player.reset()
            episode = Episode()

        observation = observation.float().to(agent.device)

        action, log_prob, value = agent.sample_action(observation)

        new_observation, reward, done, info = player.step(action)

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
            agent.write_to_tboard("Agent Outputs/Average Episode Reward", float(np.sum(episode.rewards)))

            agent.write_to_tboard(
                "League Agent Outputs/Average Probabilities", np.exp(np.mean(episode.log_probabilities))
            )

            if opponent.tag not in win_rates:
                win_rates[opponent.tag] = [int(reward > 0), 1]
            else:
                win_rates[opponent.tag][0] += int(reward > 0)
                win_rates[opponent.tag][1] += 1

            agent.write_to_tboard(
                f"League Training/{opponent.tag}",
                float(win_rates[opponent.tag][0] / win_rates[opponent.tag][1]),
            )

            opponent_name = matchmaker.choose_match(win_rates)
            _ = opponent.change_agent(opponent_name)
            history.add_episode(episode)

            if step % rollout_len == 0 and step > 0:
                history.build_dataset()
                data_loader = DataLoader(
                    history, batch_size=batch_size, shuffle=True, drop_last=True
                )
                epoch_losses = agent.learn_step(data_loader)

                for key in epoch_losses:
                    if key not in epoch_loss_count:
                        epoch_loss_count[key] = 0
                    for val in epoch_losses[key]:
                        agent.write_to_tboard(f"League Loss/{key}", val)
                        epoch_loss_count[key] += 1

                history.free_memory()

    player.complete_current_battle()


def league_play(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    matchmaker: MatchMaker,
    nb_steps: int,
    epoch_len: int,
    batch_size: int,
    args: DotDict,
    logdir: str,
    rollout_len: int,
):
    agent.save_args(args)

    for epoch in range(nb_steps // epoch_len):
        agent.save_model(agent.network, epoch, args)
        opponent.network = agent.network

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

        player.play_against(
            env_algorithm=league_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "matchmaker": matchmaker,
                "epoch_len": epoch_len,
                "batch_size": batch_size,
                "rollout_len": rollout_len,
            },
        )

    agent.save_model(agent.network, nb_steps // epoch_len, args)


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    env_player = RLPlayer(battle_format=args.battle_format, embed_battle=preprocessor.embed_battle,)

    matchmaker = MatchMaker(args.self_play_prob, args.league_play_prob, args.logdir, args.tag)

    opponent = LeaguePlayer(args.device, args.sample_moves)

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
    agent.save_model(agent.network, 0, args)

    env_player.play_against(
        env_algorithm=league_play,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "opponent": opponent,
            "matchmaker": matchmaker,
            "nb_steps": args.nb_steps,
            "epoch_len": args.epoch_len,
            "batch_size": args.batch_size,
            "args": args,
            "logdir": args.logdir,
            "rollout_len": args.rollout_len,
        },
    )


if __name__ == "__main__":
    main(parse_args())
