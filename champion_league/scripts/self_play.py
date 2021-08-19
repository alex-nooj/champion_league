import os

import numpy as np
from adept.utils.util import DotDict
from torch.utils.data import DataLoader

from champion_league.agent.opponent.league_player import LeaguePlayer
from champion_league.agent.ppo import PPOAgent
from champion_league.env.rl_player import RLPlayer
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.scripts.league_play import league_check
from champion_league.utils.parse_args import parse_args
from champion_league.utils.replay import Episode
from champion_league.utils.replay import History


def self_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    epoch_len: int,
    batch_size: int,
    rollout_len: int,
):
    """This function runs through a single epoch of self-play training. This is performed using the
    league code, where the league opponent is simply always restricted to chosing 'self' as the
    opponent.

    Parameters
    ----------
    player: RLPlayer
        The current player that is communicating with the server.
    agent: PPOAgent
        The agent that is choosing moves and handing them off to the player.
    opponent: LeaguePlayer
        The opponent that 'player' is playing against.
    epoch_len: int
        How many steps are in an epoch.
    batch_size: int
        The size of the batch during training.
    rollout_len: int
        Length of a rollout.

    Returns
    -------
    None
    """
    history = History()

    epoch_loss_count = {}

    done = True
    observation = None
    episode = Episode()
    win_rates = {}

    for step in range(epoch_len):
        if done:
            observation = player.reset()
            episode = Episode()

        observation = observation.float().to(agent.device)

        action, log_prob, value = agent.sample_action(observation)

        new_observation, reward, done, info = player.step(int(action))

        episode.append(
            observation.squeeze(),
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
                "League Agent Outputs/Average Episode Reward",
                float(np.sum(episode.rewards)),
            )

            agent.write_to_tboard(
                "League Agent Outputs/Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
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

            history.add_episode(episode)

            if len(history) > batch_size * rollout_len:
                history.build_dataset()
                data_loader = DataLoader(
                    history, batch_size, shuffle=True, drop_last=True
                )
                epoch_losses = agent.learn_step(data_loader)
                opponent.update_network(agent.network)

                for key in epoch_losses:
                    if key not in epoch_loss_count:
                        epoch_loss_count[key] = 0
                    for val in epoch_losses[key]:
                        agent.write_to_tboard(f"League Loss/{key}", val)
                        epoch_loss_count[key] += 1

                history.free_memory()

    player.complete_current_battle()


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    env_player = RLPlayer(
        battle_format=args.battle_format,
        embed_battle=preprocessor.embed_battle,
    )

    opponent = LeaguePlayer(
        device=args.device,
        network=network,
        preprocessor=preprocessor,
        sample_moves=args.sample_moves or True,
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

    opponent.change_agent("self")

    for epoch in range(args.nb_steps // args.epoch_len):
        agent.save_model(agent.network, epoch, args)

        env_player.play_against(
            env_algorithm=self_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "epoch_len": args.epoch_len,
                "batch_size": args.batch_size,
                "rollout_len": args.rollout_len,
            },
        )

    agent.save_model(agent.network, args.nb_steps // args.epoch_len, args)


if __name__ == "__main__":
    main(parse_args())
