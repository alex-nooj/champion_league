import json
import os

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
from champion_league.preprocessors import build_preprocessor_from_args, Preprocessor
from champion_league.utils.parse_args import parse_args


def league_check(player: RLPlayer, agent: PPOAgent, opponent: LeaguePlayer, logdir: str,) -> bool:
    # Grab all of the league agents
    league_agents = os.listdir(os.path.join(logdir, "league"))

    overall_wins = 0
    for league_agent in league_agents:
        _ = opponent.change_agent(os.path.join(logdir, "league", league_agent))
        current_wins = 0
        for episode in range(100):
            done = False
            observation = player.reset()
            reward = 0
            while not done:
                observation = observation.float().to(agent.device)
                action = agent.act(observation)
                observation, reward, done, _ = player.step(action)
            current_wins += int(reward > 0)
        overall_wins += int(current_wins >= 50)
    return overall_wins / len(league_agents) >= 0.75


def move_to_league(logdir: str, tag: str, epoch: int, args: DotDict, network: torch.nn.Module):
    save_dir = os.path.join(logdir, "league", f"{tag}_{epoch:07d}")
    os.mkdir(save_dir)

    with open(os.path.join(save_dir, "args.json"), "w") as fp:
        json.dump(args, fp, indent=2)

    torch.save(network.state_dict(), os.path.join(save_dir, "network.pt"))


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
    preprocessor: Preprocessor,
):
    history = History()

    epoch_loss_count = {}

    done = True
    observation = None
    episode = Episode()
    episode_ite = 0
    win_rates = {}

    opponent_name = matchmaker.choose_match(win_rates)
    if opponent_name == "self":
        opponent_name = opponent.change_agent(opponent_name, agent.network, preprocessor)
    else:
        opponent_name = opponent.change_agent(opponent_name)

    for step in tqdm(range(1, nb_steps + 1)):
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
            agent.write_to_tboard(
                "Average Episode Reward", float(np.sum(episode.rewards)), episode_ite
            )
            agent.write_to_tboard(
                "Average Probabilities", np.exp(np.mean(episode.log_probabilities)), episode_ite
            )

            if opponent_name not in win_rates:
                win_rates[opponent_name] = [int(reward > 0), 1]
            else:
                win_rates[opponent_name][0] += int(reward > 0)
                win_rates[opponent_name][1] += 1

            agent.write_to_tboard(
                f"Win Rates/{opponent_name}",
                float(win_rates[opponent_name][0] / win_rates[opponent_name][1]),
                win_rates[opponent_name][1],
            )

            opponent_name = matchmaker.choose_match(win_rates)
            if opponent_name == "self":
                opponent_name = opponent.change_agent(opponent_name, agent.network, preprocessor)
            else:
                opponent_name = opponent.change_agent(opponent_name)
            history.add_episode(episode)
            episode_ite += 1

        if step % epoch_len == 0:
            history.build_dataset()
            data_loader = DataLoader(history, batch_size=batch_size, shuffle=True, drop_last=True)
            epoch_losses = agent.learn_step(data_loader)

            for key in epoch_losses:
                if key not in epoch_loss_count:
                    epoch_loss_count[key] = 0
                for val in epoch_losses[key]:
                    agent.write_to_tboard(f"Loss/{key}", val, epoch_loss_count[key])
                    epoch_loss_count[key] += 1

            history.free_memory()

            agent.save_model(agent.network, step // epoch_len, args)
            opponent.network = agent.network
            player.complete_current_battle()

            if league_check(player, agent, opponent, logdir):
                move_to_league(logdir, agent.tag, step // epoch_len, args, agent.network)
            done = True


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    env_player = RLPlayer(battle_format=args.battle_format, embed_battle=preprocessor.embed_battle,)

    matchmaker = MatchMaker(args.self_play_prob, args.league_play_prob, args.logdir, args.tag)

    opponent = LeaguePlayer(args.sample_moves)

    agent = PPOAgent(
        device=args.gpu_id,
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
            "preprocessor": preprocessor,
        },
    )


if __name__ == "__main__":
    main(parse_args())
