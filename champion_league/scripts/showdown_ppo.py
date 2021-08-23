import numpy as np
from champion_league.utils.directory_utils import DotDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from champion_league.agent.ppo.ppo_agent import PPOAgent
from champion_league.agent.scripted.max_damage_player import MaxDamagePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.utils.parse_args import parse_args
from champion_league.utils.replay import Episode
from champion_league.utils.replay import History


def train_loop(
    player: RLPlayer,
    agent: PPOAgent,
    nb_games: int,
    nb_epochs: int,
    batch_size: int,
    save_after: int,
):
    history = History()

    epoch_loss_count = {}
    episode_ite = 0

    for ite in tqdm(range(nb_epochs)):
        if ite % save_after == 0:
            agent.save_model(agent.network, ite)

        for episode_i in range(nb_games):
            observation = player.reset()
            episode = Episode()

            done = False
            while not done:
                observation = observation.float().to(agent.device)

                action, log_probability, value = agent.sample_action(observation)

                new_observation, reward, done, info = player.step(action)

                episode.append(
                    observation=observation.squeeze(),
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=1.0,
                )

                observation = new_observation

            episode.end_episode(last_value=0)

            episode_ite += 1
            agent.write_to_tboard(
                "Average Episode Reward", float(np.sum(episode.rewards)), episode_ite
            )
            agent.write_to_tboard(
                "Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
                episode_ite,
            )

            history.add_episode(episode)

        history.build_dataset()
        data_loader = DataLoader(
            history, batch_size=batch_size, shuffle=True, drop_last=True
        )
        epoch_losses = agent.learn_step(data_loader)

        for key in epoch_losses:
            if key not in epoch_loss_count:
                epoch_loss_count[key] = 0
            for val in epoch_losses[key]:
                agent.write_to_tboard(f"Loss/{key}", val, epoch_loss_count[key])
                epoch_loss_count[key] += 1

        history.free_memory()

    agent.save_model(agent.network, 0, f"ppo_final_policy.pth")


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    env_player = RLPlayer(
        battle_format=args.battle_format,
        embed_battle=preprocessor.embed_battle,
    )

    opponent = MaxDamagePlayer(args)

    agent = PPOAgent(
        device=args.gpu_id,
        network=network,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        clip=args.clip,
        logdir=args.logdir,
        tag=args.tag,
    )

    env_player.play_against(
        env_algorithm=train_loop,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "nb_games": args.nb_games,
            "nb_epochs": args.epochs,
            "batch_size": args.batch_size,
            "save_after": args.save_after,
        },
    )


if __name__ == "__main__":
    main(parse_args())
