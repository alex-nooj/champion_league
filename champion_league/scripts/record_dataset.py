import os

from adept.utils.util import DotDict
from poke_env.player.baselines import SimpleHeuristicsPlayer
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List

from champion_league.preprocessors import Preprocessor
from poke_env.environment.battle import Battle
from tqdm import tqdm
import numpy as np

from champion_league.agent.imitation.imitation_agent import ImitationAgent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.scripted import SimpleHeuristic
from champion_league.env.rl_player import RLPlayer
from champion_league.network import build_network_from_args
from champion_league.ppo.replay import Episode, History
from champion_league.preprocessors import build_preprocessor_from_args


def parse_args() -> DotDict:
    from champion_league.config import CFG

    return DotDict(
        {k: tuple(v) if type(v) not in [int, float, str, bool] else v for k, v in CFG.items()}
    )


def identity_embedding(battle: Battle) -> Battle:
    return battle


def record_game(player: RLPlayer, preprocessor: Preprocessor,) -> Dict[str, List]:
    done = False
    episode_reward = []
    episode_states = []
    episode_action = []
    agent = SimpleHeuristic("simple_heuristic")
    observation = player.reset()
    while not done:
        action = agent.act(observation)
        embedded_battle = preprocessor.embed_battle(observation).numpy()
        observation, reward, done, _ = player.step(action)

        episode_reward.append(reward)
        episode_states.append(embedded_battle)
        episode_action.append(action)
    return {"actions": episode_action, "states": episode_states, "rewards": episode_reward}


def record_dataset(
    player: RLPlayer, preprocessor: Preprocessor, nb_games: int, save_dir: str,
):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    dataset = {}
    for _ in tqdm(range(nb_games)):
        data = record_game(player, preprocessor)
        for k, v in data.items():
            if k not in dataset:
                dataset[k] = v
            else:
                dataset[k] += v
    np.savez_compressed(os.path.join(save_dir, "dataset"), **dataset)


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    env_player = RLPlayer(battle_format=args.battle_format, embed_battle=identity_embedding,)

    env_player.play_against(
        env_algorithm=record_dataset,
        opponent=SimpleHeuristicsPlayer(),
        env_algorithm_kwargs={
            "preprocessor": preprocessor,
            "nb_games": args.nb_games,
            "save_dir": args.save_dir,
        },
    )


if __name__ == "__main__":
    main(parse_args())
