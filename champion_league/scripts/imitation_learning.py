import asyncio
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from torch import Tensor
from torch.utils.data import DataLoader

from champion_league.agent.imitation.imitation_agent import ImitationAgent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.agent.ppo import PPOAgent
from champion_league.env import OpponentPlayer
from champion_league.env import RLPlayer
from champion_league.network import build_network_from_args
from champion_league.preprocessors import build_preprocessor_from_args
from champion_league.preprocessors import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.parse_args import parse_args
from champion_league.utils.poke_set import PokeSet
from champion_league.utils.progress_bar import ProgressBar
from champion_league.utils.replay import cumulative_sum
from champion_league.utils.server_configuration import DockerServerConfiguration


def identity_embedding(battle: Battle) -> Battle:
    return battle


def record_episode(
    player: RLPlayer,
    agent: ImitationAgent,
) -> Tuple[Dict[str, List[Tensor]], List[int], List[float]]:
    observation = player.reset()
    reset = True
    states = {}
    actions = []
    rewards = []

    done = False
    while not done:
        action = agent.act(observation)
        new_observation, reward, done, _ = player.step(action)
        state = agent.embed_battle(observation, reset=reset)
        reset = False
        for key, value in state.items():
            if key not in states:
                states[key] = [value.cpu()]
            else:
                states[key].append(value.cpu())
        actions.append(action)
        rewards.append(reward / player.reward_scheme.max)
        observation = new_observation
    rewards.append(0)

    return states, actions, cumulative_sum(rewards)[:-1]


def record_epoch(
    player: RLPlayer,
    agent: ImitationAgent,
    nb_batches: int,
    batch_size: int,
    progress_bar: ProgressBar,
    train: bool,
) -> PokeSet:
    states = {}
    actions = []
    rewards = []

    while len(actions) < nb_batches * batch_size:
        progress_bar.print_bar(
            len(actions) / (nb_batches * batch_size),
            {"Samples Collected": len(actions)},
            train,
        )
        batch_states, batch_actions, batch_rewards = record_episode(player, agent)
        for key, value in batch_states.items():
            if key not in states:
                states[key] = value[::2]
            else:
                states[key] += value[::2]
        actions += batch_actions[::2]
        rewards += batch_rewards[::2]

    progress_bar.print_bar(1, {"Samples Collected": len(actions)}, train)
    states = {k: torch.stack(v) for k, v in states.items()}
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    return PokeSet(states, actions, rewards, agent.device)


def train_epoch(
    player: RLPlayer,
    agent: ImitationAgent,
    batch_size: int,
    nb_batches: int,
    progress_bar: ProgressBar,
):
    agent.training_set = DataLoader(
        record_epoch(player, agent, nb_batches, batch_size, progress_bar, True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def validation_epoch(
    player: RLPlayer,
    agent: ImitationAgent,
    batch_size: int,
    nb_batches: int,
    progress_bar: ProgressBar,
):
    agent.validation_set = DataLoader(
        record_epoch(player, agent, nb_batches, batch_size, progress_bar, False),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def imitation_learning(
    battle_format: str,
    reward_scheme: RewardScheme,
    preprocessor: Preprocessor,
    agent: ImitationAgent,
    nb_epochs: int,
    batch_size: int,
    patience: int,
    logdir: str,
    args: DotDict,
    batches_per_epoch: Optional[int] = 600,
):
    agent.save_model(agent.network, 0, args)

    progress_bar = ProgressBar(["Samples Collected"])
    progress_bar.set_epoch(0)
    player = RLPlayer(
        battle_format=battle_format,
        embed_battle=identity_embedding,
        reward_scheme=reward_scheme,
        server_configuration=DockerServerConfiguration,
    )

    opponent = OpponentPlayer.from_path(
        path=os.path.join(logdir, "league", "simple_heuristic_0"),
        device=agent.device,
    )

    player.play_against(
        env_algorithm=train_epoch,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "batch_size": batch_size,
            "nb_batches": batches_per_epoch,
            "progress_bar": progress_bar,
        },
    )

    player.play_against(
        env_algorithm=validation_epoch,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "batch_size": batch_size,
            "nb_batches": batches_per_epoch // 3,
            "progress_bar": progress_bar,
        },
    )

    progress_bar.close()

    min_val_loss = None
    fuse = patience

    for epoch in range(nb_epochs):
        _ = agent.learn_step(agent.training_set, epoch)
        validation_stats = agent.validation_step(agent.validation_set)

        if min_val_loss is None or validation_stats["Total"] < min_val_loss:
            min_val_loss = validation_stats["Total"]
            agent.save_model(agent.network, 0, args, "best_model.pt")
            fuse = patience
        else:
            fuse -= 1

        if fuse < 0:
            break
    agent.progress_bar.close()

    player = OpponentPlayer(
        RLOpponent(
            network=agent.network.eval(),
            preprocessor=preprocessor,
            device=args.device,
            sample_moves=False,
        ),
        max_concurrent_battles=100,
    )

    asyncio.get_event_loop().run_until_complete(battle(player, opponent, 100))

    print(player.n_won_battles)


async def battle(player1: OpponentPlayer, player2: OpponentPlayer, nb_battles: int):
    await player1.battle_against(player2, nb_battles)


def main(args: DotDict):
    preprocessor = build_preprocessor_from_args(args)
    args.in_shape = preprocessor.output_shape

    network = build_network_from_args(args).eval()

    agent = ImitationAgent(
        device=args.device,
        network=network,
        lr=args.lr,
        embed_battle=preprocessor.embed_battle,
        logdir=os.path.join(args.logdir, "challengers"),
        tag=args.tag,
    )

    reward_scheme = RewardScheme(args.rewards)
    imitation_learning(
        battle_format=args.battle_format,
        reward_scheme=reward_scheme,
        preprocessor=preprocessor,
        agent=agent,
        nb_epochs=args.nb_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        logdir=args.logdir,
        args=args,
        batches_per_epoch=args.batches_per_epoch,
    )


if __name__ == "__main__":
    main(parse_args())
