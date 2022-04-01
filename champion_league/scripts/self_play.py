from typing import Any
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader

from champion_league.agent.ppo import PPOAgent
from champion_league.config import parse_args
from champion_league.config.load_configs import save_args
from champion_league.env.league_player import LeaguePlayer
from champion_league.env.rl_player import RLPlayer
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.utils.agent_utils import build_network_and_preproc
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.poke_path import PokePath
from champion_league.utils.replay import History
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


def self_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    epoch_len: int,
    batch_size: int,
    rollout_len: int,
    step_counter: StepCounter,
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
    step_counter: StepCounter
        Tracks the total number of steps across each epoch.

    Returns
    -------
    None
    """
    start_step = step_counter.steps
    history = History()
    epoch_loss_count = {}
    win_rates = {}

    while True:
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)

        agent.write_to_tboard(
            "Agent Outputs/Average Episode Reward",
            float(np.sum(episode.rewards)),
        )

        agent.write_to_tboard(
            "Agent Outputs/Average Probabilities",
            np.exp(np.mean(episode.log_probabilities)),
        )

        if opponent.tag not in win_rates:
            win_rates[opponent.tag] = [int(episode.rewards[-1] > 0), 1]
        else:
            win_rates[opponent.tag][0] += int(episode.rewards[-1] > 0)
            win_rates[opponent.tag][1] += 1

        agent.write_to_tboard(
            f"League Training/{opponent.tag}",
            float(win_rates[opponent.tag][0] / win_rates[opponent.tag][1]),
        )

        history.add_episode(episode)

        if len(history) > batch_size * rollout_len:
            history.build_dataset()
            data_loader = DataLoader(history, batch_size, shuffle=True, drop_last=True)
            epoch_losses = agent.learn_step(data_loader)
            opponent.update_network(agent.network)

            for key in epoch_losses:
                if key not in epoch_loss_count:
                    epoch_loss_count[key] = 0
                for val in epoch_losses[key]:
                    agent.write_to_tboard(f"League Loss/{key}", val)
                    epoch_loss_count[key] += 1

            if step_counter.steps - start_step >= epoch_len:
                break

            history.free_memory()

    player.complete_current_battle()


def main(args: Dict[str, Any]):
    league_path = PokePath(args["logdir"], args["tag"])

    network, preprocessor = build_network_and_preproc(args)

    if "resume" in args and args["resume"]:
        network.resume(league_path.agent)

    reward_scheme = RewardScheme(rules=args["rewards"])

    agent = PPOAgent(
        device=args["device"],
        network=network,
        lr=args["lr"],
        entropy_weight=args["entropy_weight"],
        clip=args["clip"],
        league_path=league_path,
        tag=args["tag"],
    )

    step_counter = StepCounter()

    env_player = RLPlayer(
        battle_format=args["battle_format"],
        embed_battle=preprocessor.embed_battle,
        reward_scheme=reward_scheme,
        server_configuration=DockerServerConfiguration,
    )

    opponent = LeaguePlayer(
        device=args["device"],
        network=network,
        preprocessor=preprocessor,
        sample_moves=args["sample_moves"] or True,
        server_configuration=DockerServerConfiguration,
        battle_format=args["battle_format"],
    )

    opponent.change_agent("self")

    for epoch in range(args["nb_steps"] // args["epoch_len"]):
        save_args(agent_dir=league_path.agent, args=args, epoch=epoch)

        env_player.play_against(
            env_algorithm=self_epoch,
            opponent=opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "opponent": opponent,
                "epoch_len": args["epoch_len"],
                "batch_size": args["batch_size"],
                "rollout_len": args["rollout_len"],
                "step_counter": step_counter,
            },
        )
        agent.save_model(epoch, network)


if __name__ == "__main__":
    main(parse_args(__file__))
