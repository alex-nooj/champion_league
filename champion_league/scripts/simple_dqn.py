"""
Train a DQN Agent to play Gen8Random battles

Usage:
    simple_dqn [options]

General Options:
    --nb_train_episodes <int>       How many episodes to run training for [default: 2000]
    --nb_eval_episodes <int>        How many episodes to evalutate on [default: 100]
    --nb_epochs <int>               How many epochs to run [default: 1000]
    --batch_size <int>              Batch size [default: 256]
    --tag <str>                     Name of the network [default: None]
    --device <int>                  GPU to train on [default: 0]
    --logdir <str>                  Where to save the model [default: /home/alex/Documents/pokemon_agents/dqn_start/]
    --algorithm <str>               Which algorithm to use [default: DQN]
    --opponent_logdir <str>         Where to load the opponenet from [default: /home/alex/Documents/pokemon_agents/dqn_start/]
    --opponent_tag <str>            Name of the opponent [default: sampling_actions]
    --opponent_gpu <int>            GPU to load the opponent onto [default: 0]
"""
import time
from typing import Callable, Optional, Union, Tuple

import numpy as np
import torch
import torch.optim as optim
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from adept.utils.util import DotDict
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from champion_league.agent.dqn.agent import DQNAgent
from champion_league.agent.dqn.utils import (
    greedy_policy, sampling_policy, DQN,
)
from champion_league.env.rl_player import RLPlayer
from champion_league.utils.rl_utils import load_model

NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100
TARGET_UPDATE = 10
START_TIME = time.time()

def parse_args() -> Tuple[DotDict, DotDict]:
    from docopt import docopt

    args = docopt(__doc__)
    args = {k.strip("--").replace("-", "_"): v for k, v in args.items()}

    args = DotDict(args)
    args.nb_train_episodes = int(args.nb_train_episodes)
    args.nb_eval_episodes = int(args.nb_eval_episodes)
    args.nb_epochs = int(args.nb_epochs)
    args.tag = str(args.tag)
    args.device = int(args.device)
    args.logdir = str(args.logdir)
    args.algorithm = str(args.algorithm)
    args.batch_size = int(args.batch_size)
    args.opponent_gpu = int(args.opponent_gpu)

    opponent_args = DotDict({
        "logdir": args.opponent_logdir,
        "tag": args.opponent_tag,
        "gpu_id": args.opponent_gpu,
    })
    return args, opponent_args


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
            / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    name = "Name0"
    i = 0

    def choose_move(self, battle):
        self.i += 1
        self.name = f"Name{self.i}"
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(
                battle.available_moves, key=lambda move: move.base_power
            )
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(
    player,
    agent,
    args,
    opponent,
    selection_policy: Callable,
):
    print(opponent.name)
    for i_episode in range(args.nb_train_episodes):
        state = player.reset()

        done = False
        nb_steps = 0
        start_time = time.time()
        while not done:
            state = state.to(args.device).double()
            action = agent.choose_move(state.view(1, -1))

            next_state, reward, done, info = player.step(action.item())
            reward = torch.tensor([reward], device=args.device)

            agent.memory.push(
                state, action, next_state.double(), reward
            )
            state = next_state

            agent.learn_step()
            nb_steps += 1
        end_time = time.time()
        print(f"{i_episode}: {nb_steps/(end_time - start_time): 0.3f} steps/sec, REWARD: {int(reward[0])}")

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    player.complete_current_battle()


def dqn_evaluation(player, policy_net, args, epoch, opponent):
    # Reset battle statistics
    print(f"STARTUP TIME: {time.time() - START_TIME}")

    player.reset_battles()
    for i_episode in range(args.nb_eval_episodes):
        state = player.reset()
        done = False
        while not done:
            print(opponent.name)
            # state = torch.from_numpy(state)
            state = torch.reshape(state, (1, -1)).to(args.device).double()
            action = policy_net(state.float()).max(1)[1].view(1, 1)
            state, reward, done, info = player.step(action.item())

    print(
        "Epoch %d: %0.3f win rate" % (epoch, player.n_won_battles / args.nb_eval_episodes)
    )


def main(args: DotDict, opponent_args: DotDict):
    env_player = RLPlayer(
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration(args.tag, "test"),
    )

    # Output dimension
    args.nb_actions = len(env_player.action_space)

    second_opponent = MaxDamagePlayer(
        battle_format="gen8randombattle",
        player_configuration=PlayerConfiguration("max_damage", "test"),
    )

    agent = DQNAgent(args, PlayerConfiguration("whatever", "test"))

    for epoch in range(args.nb_epochs):
        env_player.play_against(
            env_algorithm=dqn_evaluation,
            opponent=second_opponent,
            env_algorithm_kwargs={
                "policy_net": agent.policy_net.eval().to(args.device),
                "args": args,
                "epoch": epoch,
                "opponent": second_opponent
            }
        )

        env_player.play_against(
            env_algorithm=dqn_training,
            opponent=second_opponent,
            env_algorithm_kwargs={
                "agent": agent,
                "args": args,
                "selection_policy": sampling_policy,
                "opponent": second_opponent
            }
        )
        agent.save_model(agent.network, epoch)


if __name__ == "__main__":
    args, opponent_args = parse_args()
    main(args, opponent_args)
