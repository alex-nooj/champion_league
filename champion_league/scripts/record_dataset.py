import os
from typing import Dict
from typing import List

import numpy as np
import torch
from poke_env.environment.battle import Battle
from poke_env.player.baselines import SimpleHeuristicsPlayer
from tqdm import tqdm

from champion_league.agent.scripted import SimpleHeuristic
from champion_league.env.rl_player import RLPlayer
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.preprocessors import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.parse_args import parse_args
from champion_league.utils.replay import cumulative_sum
from champion_league.utils.server_configuration import DockerServerConfiguration


class NoDynamaxingPlayer(SimpleHeuristicsPlayer):
    def _should_dynamax(self, *args, **kwargs) -> bool:
        return False


@torch.no_grad()
def record_game(
    player: RLPlayer, preprocessor: Preprocessor, gamma: float
) -> Dict[str, List]:
    done = False
    episode_reward = []
    episode_states = []
    episode_action = []
    agent = SimpleHeuristic("simple_heuristic")
    observation = player.reset()
    while not done:
        action = agent.act(observation)
        embedded_battle = preprocessor.embed_battle(observation).cpu().numpy()
        observation, reward, done, _ = player.step(action)

        episode_reward.append(reward / 6.0)
        episode_states.append(embedded_battle)
        episode_action.append(action)
    episode_reward.append(0)

    return {
        "actions": episode_action,
        "states": episode_states,
        "rewards": cumulative_sum(episode_reward, gamma)[:-1],
    }


def record_dataset(
    player: RLPlayer,
    preprocessor: Preprocessor,
    nb_games: int,
    save_dir: str,
    gamma: float,
):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    dataset = {}
    for _ in tqdm(range(nb_games)):
        data = record_game(player, preprocessor, gamma)
        for k, v in data.items():
            if k not in dataset:
                dataset[k] = v
            else:
                dataset[k] += v
    np.savez_compressed(os.path.join(save_dir, "dataset"), **dataset)


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    env_player = RLPlayer(
        battle_format=args.battle_format,
        embed_battle=identity_embedding,
        reward_scheme=RewardScheme(rules=args.rewards),
        server_configuration=DockerServerConfiguration,
    )

    env_player.play_against(
        env_algorithm=record_dataset,
        opponent=NoDynamaxingPlayer(),
        env_algorithm_kwargs={
            "preprocessor": preprocessor,
            "nb_games": args.nb_games,
            "save_dir": args.save_dir,
            "gamma": args.gamma,
        },
    )


if __name__ == "__main__":
    main(parse_args())
