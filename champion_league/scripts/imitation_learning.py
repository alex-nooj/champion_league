import asyncio
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import torch
from poke_env.environment.battle import Battle
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

from champion_league.agent.imitation.imitation_agent import ImitationAgent
from champion_league.agent.opponent.rl_opponent import RLOpponent
from champion_league.config import parse_args
from champion_league.env import OpponentPlayer
from champion_league.env import RLPlayer
from champion_league.preprocessors import Preprocessor
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.team_builder import AgentTeamBuilder
from champion_league.utils.agent_utils import build_network_and_preproc
from champion_league.utils.poke_path import PokePath
from champion_league.utils.poke_set import PokeSet
from champion_league.utils.progress_bar import ProgressBar
from champion_league.utils.replay import cumulative_sum
from champion_league.utils.server_configuration import DockerServerConfiguration


def identity_embedding(battle: Battle) -> Battle:
    """Pass-through embedding function.

    Parameters
    ----------
    battle: Battle
        The current state of the environment.

    Returns
    -------
    Battle
        The current state of the environment.
    """
    return battle


async def agent_battle(
    player1: OpponentPlayer, player2: OpponentPlayer, nb_battles: int
):
    """Function for battling between two agents.

    Parameters
    ----------
    player1
        The player that will issue the challenges.
    player2
        The player that will accept the challenges.
    nb_battles
        The number of battles to perform.

    Returns
    -------
    None
    """
    await player1.battle_against(player2, nb_battles)


def record_episode(
    player: RLPlayer,
    agent: ImitationAgent,
) -> Tuple[Dict[str, List[Tensor]], List[int], List[float]]:
    """Records a single battle.

    Parameters
    ----------
    player
        The environment that is communicating with Showdown.
    agent
        The agent that is playing the game.

    Returns
    -------
    Tuple[Dict[str, List[Tensor]], List[int], List[float]]
        Tuple containing the observations, actions, and rewards-to-go, respectively.
    """
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
    batch_size: int,
    nb_batches: int,
    progress_bar: ProgressBar,
    train: bool,
) -> PokeSet:
    """Records all of the state, action, and value pairs from the agent we'd like to imitate.

    Parameters
    ----------
    player
        The environment that is communicating with Showdown.
    agent
        The agent that is playing the game.
    batch_size
        The number of samples to include in one batch.
    nb_batches
        The number of batches to collect.
    progress_bar
        Used to print the progress so users don't go insane.
    train
        Whether this is a training epoch.

    Returns
    -------
    PokeSet
        The recorded dataset (Subclass of PyTorch's Dataset class).
    """
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
    """Records the training dataset for the agent.

    Parameters
    ----------
    player
        The environment that is communicating with Showdown.
    agent
        The agent that is playing the game.
    batch_size
        The number of samples to include in one batch.
    nb_batches
        The number of batches to collect.
    progress_bar
        Used to print the progress so users don't go insane.

    Returns
    -------
    None
    """
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
    """Records the validation dataset for the agent.

    Parameters
    ----------
    player
        The environment that is communicating with Showdown.
    agent
        The agent that is playing the game.
    batch_size
        The number of samples to include in one batch.
    nb_batches
        The number of batches to collect.
    progress_bar
        Used to print the progress so users don't go insane.

    Returns
    -------
    None
    """
    agent.validation_set = DataLoader(
        record_epoch(player, agent, nb_batches, batch_size, progress_bar, False),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def imitation_learning(
    preprocessor: Preprocessor, network: nn.Module, league_path, args: Dict[str, Any]
):
    """The main loop for performing supervised learning between a network and a trained agent.

    Parameters
    ----------
    league_path
    preprocessor
        The preprocessor that this agent will be using to convert Battle objects to tensors.
    network
        The network that will be training.
    args
        Hyperparameters used for training. MUST CONTAIN:
        - batch_size: int
        - battle_format: str
        - batches_per_epoch
        - device: int
        - logdir: str
        - nb_epochs: int
        - patience: int
        - rewards: Dict[str, float]
        - tag: str

    Returns
    -------
    None
    """

    agent = ImitationAgent(
        device=args["device"],
        network=network,
        lr=args["lr"],
        embed_battle=preprocessor.embed_battle,
        league_path=league_path,
        tag=args["tag"],
    )

    team_builder = AgentTeamBuilder(league_path.agent)

    agent.save_model(0, agent.network)
    reward_scheme = RewardScheme(args["rewards"])

    progress_bar = ProgressBar(["Samples Collected"])
    progress_bar.set_epoch(0)

    player = RLPlayer(
        battle_format=args["battle_format"],
        embed_battle=identity_embedding,
        reward_scheme=reward_scheme,
        server_configuration=DockerServerConfiguration,
        team=team_builder,
    )

    opponent = OpponentPlayer.from_path(
        path=league_path.league / "simple_heuristic_0",
        device=agent.device,
        team=AgentTeamBuilder(),
        battle_format=args["battle_format"],
    )

    player.play_against(
        env_algorithm=train_epoch,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "batch_size": args["batch_size"],
            "nb_batches": args["batches_per_epoch"],
            "progress_bar": progress_bar,
        },
    )

    player.play_against(
        env_algorithm=validation_epoch,
        opponent=opponent,
        env_algorithm_kwargs={
            "agent": agent,
            "batch_size": args["batch_size"],
            "nb_batches": args["batches_per_epoch"] // 3,
            "progress_bar": progress_bar,
        },
    )

    progress_bar.close()

    min_val_loss = None
    fuse = args["patience"]

    for epoch in range(args["nb_epochs"]):
        _ = agent.learn_step(agent.training_set, epoch)
        validation_stats = agent.validation_step(agent.validation_set)

        if min_val_loss is None or validation_stats["Total"] < min_val_loss:
            min_val_loss = validation_stats["Total"]
            agent.save_model(0, agent.network)
            fuse = args["patience"]
        else:
            fuse -= 1

        if fuse < 0:
            break
    agent.progress_bar.close()

    player = OpponentPlayer(
        RLOpponent(
            network=agent.network.eval(),
            preprocessor=preprocessor,
            device=args["device"],
            sample_moves=False,
        ),
        max_concurrent_battles=100,
        team=team_builder,
        battle_format=args["battle_format"],
    )

    asyncio.get_event_loop().run_until_complete(agent_battle(player, opponent, 100))

    print(player.n_won_battles)


def main(args: Dict[str, Any]):
    league_path = PokePath(args["logdir"], args["tag"])

    network, preprocessor = build_network_and_preproc(args)
    if "resume" in args and args["resume"]:
        network.resume(league_path.agent)

    imitation_learning(
        preprocessor=preprocessor, network=network, league_path=league_path, args=args
    )


if __name__ == "__main__":
    main(parse_args(__file__))
