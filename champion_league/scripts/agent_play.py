from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
from poke_env.teambuilder.teambuilder import Teambuilder
from torch import nn
from torch.utils.data import DataLoader

from champion_league.agent.ppo import PPOAgent
from champion_league.config import parse_args
from champion_league.config.load_configs import save_args
from champion_league.env import LeaguePlayer
from champion_league.env import RLPlayer
from champion_league.network import NETWORKS
from champion_league.preprocessors import Preprocessor
from champion_league.preprocessors import PREPROCESSORS
from champion_league.reward.reward_scheme import RewardScheme
from champion_league.teams.team_builder import load_team_from_file
from champion_league.utils.collect_episode import collect_episode
from champion_league.utils.replay import History
from champion_league.utils.server_configuration import DockerServerConfiguration
from champion_league.utils.step_counter import StepCounter


class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        team = load_team_from_file(
            "/home/anewgent/Documents/pokemon_trainers/challengers/quick_test/"
        )
        parsed_team = self.parse_showdown_team(team)
        self.team = self.join_team(parsed_team)

    def yield_team(self):
        return self.team


def agent_epoch(
    player: RLPlayer,
    agent: PPOAgent,
    opponent: LeaguePlayer,
    batch_size: int,
    rollout_len: int,
    epoch_len: int,
    step_counter: StepCounter,
) -> None:
    start_step = step_counter.steps
    history = History()

    while step_counter.steps - start_step < epoch_len or len(history) != 0:
        episode = collect_episode(player=player, agent=agent, step_counter=step_counter)

        agent.update_winrates(opponent.tag, int(episode.rewards[-1] > 0))

        agent.write_to_tboard(
            f"Agent Play/{opponent.tag}", np.mean(agent.win_rates[opponent.tag])
        )

        history.add_episode(episode)

        if len(history) > batch_size * rollout_len:
            history.build_dataset()
            data_loader = DataLoader(
                history,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            epoch_losses = agent.learn_step(data_loader)

            for k, v in epoch_losses.items():
                agent.write_to_tboard(f"League Loss/{k}", float(np.mean(v)))

            history.free_memory()


custom_builder = RandomTeamFromPool([])


def agent_play(preprocessor: Preprocessor, network: nn.Module, args: Dict[str, Any]):

    agent_dir = Path(args["logdir"], "challengers", args["tag"])
    agent = PPOAgent(
        device=args["device"],
        network=network,
        lr=args["lr"],
        entropy_weight=args["entropy_weight"],
        clip=args["clip"],
        challenger_dir=Path(args["logdir"], "challengers"),
        tag=args["tag"],
    )
    save_args(agent_dir, 0, args)

    step_counter = StepCounter()

    starting_epoch = 0

    for opponent_path in args["opponents"]:
        for epoch in range(
            starting_epoch, args["nb_steps"] // args["epoch_len"] + starting_epoch
        ):
            print(f"\nAgent {opponent_path.rsplit('/')[-1]} - Epoch {epoch:3d}\n")

            agent.save_model(agent_dir, epoch, agent.network)

            opponent = LeaguePlayer(
                device=args["device"],
                network=network,
                preprocessor=preprocessor,
                sample_moves=False,
                team=custom_builder,
                battle_format=args["battle_format"],
            )
            opponent.change_agent(opponent_path)

            player = RLPlayer(
                battle_format=args["battle_format"],
                embed_battle=preprocessor.embed_battle,
                reward_scheme=RewardScheme(args["rewards"]),
                server_configuration=DockerServerConfiguration,
                team=custom_builder,
            )

            player.play_against(
                env_algorithm=agent_epoch,
                opponent=opponent,
                env_algorithm_kwargs={
                    "agent": agent,
                    "opponent": opponent,
                    "batch_size": args["batch_size"],
                    "rollout_len": args["rollout_len"],
                    "epoch_len": args["epoch_len"],
                    "step_counter": step_counter,
                },
            )

            if np.mean(agent.win_rates[opponent.tag]) > 0.7:
                break


def main(args: Dict[str, Any]):
    preprocessor = PREPROCESSORS[args["preprocessor"]](
        args["device"], **args[args["preprocessor"]]
    )
    network = (
        NETWORKS[args["network"]](
            nb_actions=args["nb_actions"],
            in_shape=preprocessor.output_shape,
            **args[args["network"]],
        )
        .eval()
        .to(args["device"])
    )

    agent_play(preprocessor=preprocessor, network=network, args=args)


if __name__ == "__main__":
    main(parse_args(__file__))
