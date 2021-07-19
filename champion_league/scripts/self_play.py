import numpy as np
from adept.utils.util import DotDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from champion_league.agent.base.base_agent import BaseAgent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.ppo import PPOAgent
from champion_league.env.rl_player import RLPlayer
from champion_league.network import build_network_from_args
from champion_league.ppo.replay import Episode
from champion_league.ppo.replay import History
from champion_league.preprocessors import build_preprocessor_from_args


def parse_args() -> DotDict:
    from champion_league.config import CFG

    return DotDict(
        {k: tuple(v) if type(v) not in [int, float, str, bool] else v for k, v in CFG.items()}
    )


def self_play(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: RLOpponent,
    nb_steps: int,
    epoch_len: int,
    batch_size: int,
    nb_eval_episodes: Optional[int] = 100,
):
    history = History()

    epoch_loss_count = {}

    done = True
    observation = None
    episode = Episode()
    episode_ite = 0
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

            agent.save_model(agent.network, step // epoch_len)
            opponent.network = agent.network
            player.complete_current_battle()
            opponent.eval()
            total_reward = 0

            for episode in range(nb_eval_episodes):
                done = False
                observation = player.reset()
                while not done:
                    observation = observation.float().to(agent.device)
                    action = agent.act(observation)
                    observation, reward, done, _ = player.step(action)
                total_reward += reward
            agent.write_to_tboard(
                "Validation Results/Simple Heuristic",
                total_reward / nb_eval_episodes,
                step // epoch_len,
            )
            opponent.train()
            done = True


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    env_player = RLPlayer(battle_format=args.battle_format, embed_battle=preprocessor.embed_battle,)

    opponent = RLOpponent(network, preprocessor, True)

    agent = PPOAgent(
        device=args.gpu_id,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=args.logdir,
        tag=args.tag,
    )

    agent.save_args(args)

    env_player.play_against(
        env_algorithm=self_play,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "opponent": opponent,
            "nb_steps": args.nb_steps,
            "epoch_len": args.epoch_len,
            "batch_size": args.batch_size,
        },
    )


if __name__ == "__main__":
    main(parse_args())
